"""
Long-term memory system for Agent Scrivener.

Provides knowledge base storage, intelligent memory pruning,
and memory search and retrieval capabilities.
"""

import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass
import json
import hashlib

from pydantic import BaseModel, Field
from ..models.core import Insight, Source, ExtractedArticle, AcademicPaper
from ..utils.logging import get_logger
from ..utils.error_handler import ErrorHandler

logger = get_logger(__name__)


class MemoryType(str, Enum):
    """Types of memory entries."""
    INSIGHT = "insight"
    SOURCE = "source"
    FACT = "fact"
    PATTERN = "pattern"
    RELATIONSHIP = "relationship"


class MemoryImportance(str, Enum):
    """Importance levels for memory entries."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MemoryEntry(BaseModel):
    """Individual memory entry in the knowledge base."""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType
    content: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    importance: MemoryImportance = MemoryImportance.MEDIUM
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    source_references: List[str] = Field(default_factory=list)  # Source URLs or IDs
    related_entries: List[str] = Field(default_factory=list)  # Related memory entry IDs
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    decay_factor: float = Field(default=1.0, ge=0.0, le=1.0)
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def calculate_relevance_score(self, current_time: Optional[datetime] = None) -> float:
        """
        Calculate relevance score based on importance, confidence, recency, and access patterns.
        
        Args:
            current_time: Current time for calculations (defaults to now)
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Base score from importance and confidence
        importance_weights = {
            MemoryImportance.CRITICAL: 1.0,
            MemoryImportance.HIGH: 0.8,
            MemoryImportance.MEDIUM: 0.6,
            MemoryImportance.LOW: 0.4
        }
        
        base_score = (importance_weights[self.importance] + self.confidence_score) / 2
        
        # Recency factor (decay over time)
        time_since_created = (current_time - self.created_at).total_seconds()
        time_since_accessed = (current_time - self.last_accessed).total_seconds()
        
        # Decay based on time since creation and last access
        creation_decay = max(0.1, 1.0 - (time_since_created / (30 * 24 * 3600)))  # 30 days
        access_decay = max(0.1, 1.0 - (time_since_accessed / (7 * 24 * 3600)))    # 7 days
        
        # Access frequency boost
        access_boost = min(1.0, 1.0 + (self.access_count * 0.1))
        
        # Apply decay factor
        final_score = base_score * creation_decay * access_decay * access_boost * self.decay_factor
        
        return min(1.0, max(0.0, final_score))
    
    def get_content_hash(self) -> str:
        """Get hash of content for deduplication."""
        content_str = f"{self.memory_type}:{self.content}:{':'.join(sorted(self.tags))}"
        return hashlib.md5(content_str.encode()).hexdigest()


class MemoryQuery(BaseModel):
    """Query for memory search."""
    query_text: str = Field(..., min_length=1)
    memory_types: Optional[List[MemoryType]] = None
    tags: Optional[List[str]] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_importance: Optional[MemoryImportance] = None
    max_age_days: Optional[int] = Field(None, gt=0)
    limit: int = Field(default=10, gt=0, le=100)
    include_related: bool = False


class MemorySearchResult(BaseModel):
    """Result from memory search."""
    entry: MemoryEntry
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    match_reasons: List[str] = Field(default_factory=list)


class MemoryStatistics(BaseModel):
    """Statistics about the memory store."""
    total_entries: int = 0
    entries_by_type: Dict[str, int] = Field(default_factory=dict)
    entries_by_importance: Dict[str, int] = Field(default_factory=dict)
    average_confidence: float = 0.0
    oldest_entry_age_days: Optional[float] = None
    most_accessed_entry_id: Optional[str] = None
    total_storage_size_mb: float = 0.0
    pruning_candidates: int = 0


class MemoryStore:
    """
    Long-term memory system for storing and retrieving research insights.
    
    Provides intelligent storage, search, and pruning capabilities for
    building a persistent knowledge base.
    """
    
    def __init__(self, storage_path: Optional[str] = None, max_entries: int = 10000):
        """
        Initialize memory store.
        
        Args:
            storage_path: Path to memory storage directory
            max_entries: Maximum number of entries to store
        """
        self.storage_path = storage_path or "./data/memory"
        self.max_entries = max_entries
        self.entries: Dict[str, MemoryEntry] = {}
        self.content_hashes: Dict[str, str] = {}  # hash -> entry_id mapping
        self.tag_index: Dict[str, Set[str]] = {}  # tag -> set of entry_ids
        self.type_index: Dict[MemoryType, Set[str]] = {}  # type -> set of entry_ids
        self.error_handler = ErrorHandler()
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the memory store."""
        import os
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Load existing memories
        await self._load_memories()
        
        logger.info(f"Initialized memory store with {len(self.entries)} entries")
    
    async def store_insight(self, insight: Insight, importance: MemoryImportance = MemoryImportance.MEDIUM) -> str:
        """
        Store an insight in long-term memory.
        
        Args:
            insight: The insight to store
            importance: Importance level of the insight
            
        Returns:
            Entry ID of the stored insight
        """
        entry = MemoryEntry(
            memory_type=MemoryType.INSIGHT,
            content=insight.summary,
            metadata={
                "topic": insight.topic,
                "supporting_evidence": insight.supporting_evidence,
                "original_insight_data": insight.model_dump()
            },
            tags=insight.tags,
            importance=importance,
            confidence_score=insight.confidence_score,
            source_references=[str(source.url) for source in insight.related_sources]
        )
        
        return await self.add_entry(entry)
    
    async def store_source(self, source: Source, importance: MemoryImportance = MemoryImportance.LOW) -> str:
        """
        Store a source in long-term memory.
        
        Args:
            source: The source to store
            importance: Importance level of the source
            
        Returns:
            Entry ID of the stored source
        """
        entry = MemoryEntry(
            memory_type=MemoryType.SOURCE,
            content=f"{source.title} - {source.author or 'Unknown Author'}",
            metadata={
                "url": str(source.url),
                "title": source.title,
                "author": source.author,
                "publication_date": source.publication_date.isoformat() if source.publication_date else None,
                "source_type": source.source_type.value,
                "original_source_data": source.model_dump()
            },
            tags=[source.source_type.value],
            importance=importance,
            confidence_score=0.8,  # Default confidence for sources
            source_references=[str(source.url)]
        )
        
        return await self.add_entry(entry)
    
    async def store_fact(self, 
                        fact: str, 
                        sources: List[str], 
                        confidence: float,
                        tags: Optional[List[str]] = None,
                        importance: MemoryImportance = MemoryImportance.MEDIUM) -> str:
        """
        Store a fact in long-term memory.
        
        Args:
            fact: The fact to store
            sources: Source references for the fact
            confidence: Confidence score for the fact
            tags: Optional tags for categorization
            importance: Importance level of the fact
            
        Returns:
            Entry ID of the stored fact
        """
        entry = MemoryEntry(
            memory_type=MemoryType.FACT,
            content=fact,
            tags=tags or [],
            importance=importance,
            confidence_score=confidence,
            source_references=sources
        )
        
        return await self.add_entry(entry)
    
    async def add_entry(self, entry: MemoryEntry) -> str:
        """
        Add a memory entry to the store.
        
        Args:
            entry: The memory entry to add
            
        Returns:
            Entry ID of the added entry
        """
        async with self._lock:
            # Check for duplicates
            content_hash = entry.get_content_hash()
            if content_hash in self.content_hashes:
                existing_id = self.content_hashes[content_hash]
                logger.debug(f"Duplicate content detected, updating existing entry {existing_id}")
                
                # Update existing entry
                existing_entry = self.entries[existing_id]
                existing_entry.confidence_score = max(existing_entry.confidence_score, entry.confidence_score)
                existing_entry.importance = max(existing_entry.importance, entry.importance, key=lambda x: list(MemoryImportance).index(x))
                existing_entry.source_references.extend(entry.source_references)
                existing_entry.source_references = list(set(existing_entry.source_references))  # Remove duplicates
                existing_entry.update_access()
                
                await self._save_entry(existing_entry)
                return existing_id
            
            # Check if we need to prune before adding
            if len(self.entries) >= self.max_entries:
                await self._prune_memories(target_count=int(self.max_entries * 0.8))
            
            # Add new entry
            self.entries[entry.entry_id] = entry
            self.content_hashes[content_hash] = entry.entry_id
            
            # Update indexes
            self._update_indexes(entry)
            
            # Save to disk
            await self._save_entry(entry)
            
            logger.debug(f"Added memory entry {entry.entry_id} of type {entry.memory_type}")
            return entry.entry_id
    
    async def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Get a memory entry by ID.
        
        Args:
            entry_id: The entry identifier
            
        Returns:
            Memory entry if found, None otherwise
        """
        async with self._lock:
            entry = self.entries.get(entry_id)
            if entry:
                entry.update_access()
                await self._save_entry(entry)
            return entry
    
    async def search_memories(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """
        Search memories based on query criteria.
        
        Args:
            query: Search query parameters
            
        Returns:
            List of matching memory entries with relevance scores
        """
        async with self._lock:
            candidates = set(self.entries.keys())
            
            # Filter by memory types
            if query.memory_types:
                type_candidates = set()
                for memory_type in query.memory_types:
                    type_candidates.update(self.type_index.get(memory_type, set()))
                candidates &= type_candidates
            
            # Filter by tags
            if query.tags:
                tag_candidates = set()
                for tag in query.tags:
                    tag_candidates.update(self.tag_index.get(tag, set()))
                candidates &= tag_candidates
            
            # Filter by confidence and importance
            filtered_candidates = []
            current_time = datetime.now()
            
            for entry_id in candidates:
                entry = self.entries[entry_id]
                
                # Confidence filter
                if query.min_confidence and entry.confidence_score < query.min_confidence:
                    continue
                
                # Importance filter
                if query.min_importance:
                    importance_levels = list(MemoryImportance)
                    if importance_levels.index(entry.importance) < importance_levels.index(query.min_importance):
                        continue
                
                # Age filter
                if query.max_age_days:
                    age_days = (current_time - entry.created_at).days
                    if age_days > query.max_age_days:
                        continue
                
                filtered_candidates.append(entry)
            
            # Calculate relevance scores and rank
            results = []
            for entry in filtered_candidates:
                relevance_score = self._calculate_query_relevance(entry, query)
                match_reasons = self._get_match_reasons(entry, query)
                
                results.append(MemorySearchResult(
                    entry=entry,
                    relevance_score=relevance_score,
                    match_reasons=match_reasons
                ))
            
            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Apply limit
            results = results[:query.limit]
            
            # Update access counts for returned entries
            for result in results:
                result.entry.update_access()
                await self._save_entry(result.entry)
            
            # Include related entries if requested
            if query.include_related:
                await self._include_related_entries(results)
            
            logger.debug(f"Memory search returned {len(results)} results for query: {query.query_text[:50]}...")
            return results
    
    async def get_related_entries(self, entry_id: str, max_related: int = 5) -> List[MemoryEntry]:
        """
        Get entries related to a specific entry.
        
        Args:
            entry_id: The entry to find relations for
            max_related: Maximum number of related entries to return
            
        Returns:
            List of related memory entries
        """
        entry = await self.get_entry(entry_id)
        if not entry:
            return []
        
        related_entries = []
        
        # Get explicitly related entries
        for related_id in entry.related_entries:
            related_entry = self.entries.get(related_id)
            if related_entry:
                related_entries.append(related_entry)
        
        # Find entries with similar tags
        if len(related_entries) < max_related:
            similar_entries = await self._find_similar_entries(entry, max_related - len(related_entries))
            related_entries.extend(similar_entries)
        
        return related_entries[:max_related]
    
    async def add_relationship(self, entry_id1: str, entry_id2: str) -> bool:
        """
        Add a relationship between two memory entries.
        
        Args:
            entry_id1: First entry ID
            entry_id2: Second entry ID
            
        Returns:
            True if relationship was added, False otherwise
        """
        async with self._lock:
            entry1 = self.entries.get(entry_id1)
            entry2 = self.entries.get(entry_id2)
            
            if not entry1 or not entry2:
                return False
            
            # Add bidirectional relationship
            if entry_id2 not in entry1.related_entries:
                entry1.related_entries.append(entry_id2)
                await self._save_entry(entry1)
            
            if entry_id1 not in entry2.related_entries:
                entry2.related_entries.append(entry_id1)
                await self._save_entry(entry2)
            
            logger.debug(f"Added relationship between {entry_id1} and {entry_id2}")
            return True
    
    async def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            entry_id: The entry to delete
            
        Returns:
            True if entry was deleted, False otherwise
        """
        async with self._lock:
            entry = self.entries.get(entry_id)
            if not entry:
                return False
            
            # Remove from indexes
            self._remove_from_indexes(entry)
            
            # Remove content hash
            content_hash = entry.get_content_hash()
            self.content_hashes.pop(content_hash, None)
            
            # Remove from entries
            del self.entries[entry_id]
            
            # Remove relationships
            for other_entry in self.entries.values():
                if entry_id in other_entry.related_entries:
                    other_entry.related_entries.remove(entry_id)
                    await self._save_entry(other_entry)
            
            # Delete from disk
            await self._delete_entry_file(entry_id)
            
            logger.debug(f"Deleted memory entry {entry_id}")
            return True
    
    async def prune_memories(self, 
                           max_age_days: Optional[int] = None,
                           min_relevance_score: Optional[float] = None,
                           target_count: Optional[int] = None) -> int:
        """
        Prune old or irrelevant memories.
        
        Args:
            max_age_days: Remove entries older than this many days
            min_relevance_score: Remove entries with relevance below this threshold
            target_count: Target number of entries to keep (removes lowest relevance)
            
        Returns:
            Number of entries pruned
        """
        return await self._prune_memories(max_age_days, min_relevance_score, target_count)
    
    async def get_statistics(self) -> MemoryStatistics:
        """
        Get statistics about the memory store.
        
        Returns:
            Memory store statistics
        """
        async with self._lock:
            stats = MemoryStatistics()
            
            if not self.entries:
                return stats
            
            stats.total_entries = len(self.entries)
            
            # Count by type
            for entry in self.entries.values():
                entry_type = entry.memory_type.value
                stats.entries_by_type[entry_type] = stats.entries_by_type.get(entry_type, 0) + 1
                
                importance = entry.importance.value
                stats.entries_by_importance[importance] = stats.entries_by_importance.get(importance, 0) + 1
            
            # Calculate average confidence
            total_confidence = sum(entry.confidence_score for entry in self.entries.values())
            stats.average_confidence = total_confidence / len(self.entries)
            
            # Find oldest entry
            oldest_entry = min(self.entries.values(), key=lambda x: x.created_at)
            age_delta = datetime.now() - oldest_entry.created_at
            stats.oldest_entry_age_days = age_delta.total_seconds() / (24 * 3600)
            
            # Find most accessed entry
            most_accessed = max(self.entries.values(), key=lambda x: x.access_count)
            stats.most_accessed_entry_id = most_accessed.entry_id
            
            # Calculate storage size (approximate)
            total_size = sum(len(entry.content.encode('utf-8')) for entry in self.entries.values())
            stats.total_storage_size_mb = total_size / (1024 * 1024)
            
            # Count pruning candidates (low relevance entries)
            current_time = datetime.now()
            pruning_candidates = 0
            for entry in self.entries.values():
                if entry.calculate_relevance_score(current_time) < 0.3:
                    pruning_candidates += 1
            stats.pruning_candidates = pruning_candidates
            
            return stats
    
    def _update_indexes(self, entry: MemoryEntry):
        """Update search indexes for an entry."""
        # Update tag index
        for tag in entry.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(entry.entry_id)
        
        # Update type index
        if entry.memory_type not in self.type_index:
            self.type_index[entry.memory_type] = set()
        self.type_index[entry.memory_type].add(entry.entry_id)
    
    def _remove_from_indexes(self, entry: MemoryEntry):
        """Remove an entry from search indexes."""
        # Remove from tag index
        for tag in entry.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(entry.entry_id)
                if not self.tag_index[tag]:
                    del self.tag_index[tag]
        
        # Remove from type index
        if entry.memory_type in self.type_index:
            self.type_index[entry.memory_type].discard(entry.entry_id)
            if not self.type_index[entry.memory_type]:
                del self.type_index[entry.memory_type]
    
    def _calculate_query_relevance(self, entry: MemoryEntry, query: MemoryQuery) -> float:
        """Calculate relevance score for a query."""
        base_relevance = entry.calculate_relevance_score()
        
        # Text matching score (simple keyword matching)
        query_words = set(query.query_text.lower().split())
        content_words = set(entry.content.lower().split())
        tag_words = set(tag.lower() for tag in entry.tags)
        
        # Calculate word overlap
        content_overlap = len(query_words & content_words) / max(len(query_words), 1)
        tag_overlap = len(query_words & tag_words) / max(len(query_words), 1)
        
        text_relevance = max(content_overlap, tag_overlap)
        
        # Combine scores
        final_score = (base_relevance * 0.7) + (text_relevance * 0.3)
        
        return min(1.0, final_score)
    
    def _get_match_reasons(self, entry: MemoryEntry, query: MemoryQuery) -> List[str]:
        """Get reasons why an entry matches a query."""
        reasons = []
        
        # Check text matches
        query_words = set(query.query_text.lower().split())
        content_words = set(entry.content.lower().split())
        tag_words = set(tag.lower() for tag in entry.tags)
        
        content_matches = query_words & content_words
        tag_matches = query_words & tag_words
        
        if content_matches:
            reasons.append(f"Content matches: {', '.join(content_matches)}")
        
        if tag_matches:
            reasons.append(f"Tag matches: {', '.join(tag_matches)}")
        
        # Check type matches
        if query.memory_types and entry.memory_type in query.memory_types:
            reasons.append(f"Type match: {entry.memory_type.value}")
        
        # Check importance
        if entry.importance in [MemoryImportance.CRITICAL, MemoryImportance.HIGH]:
            reasons.append(f"High importance: {entry.importance.value}")
        
        return reasons
    
    async def _find_similar_entries(self, entry: MemoryEntry, max_count: int) -> List[MemoryEntry]:
        """Find entries similar to the given entry."""
        similar_entries = []
        entry_tags = set(entry.tags)
        
        for other_entry in self.entries.values():
            if other_entry.entry_id == entry.entry_id:
                continue
            
            # Calculate similarity based on tags
            other_tags = set(other_entry.tags)
            tag_overlap = len(entry_tags & other_tags)
            
            if tag_overlap > 0:
                similarity_score = tag_overlap / len(entry_tags | other_tags)
                if similarity_score > 0.3:  # Threshold for similarity
                    similar_entries.append((other_entry, similarity_score))
        
        # Sort by similarity and return top entries
        similar_entries.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in similar_entries[:max_count]]
    
    async def _include_related_entries(self, results: List[MemorySearchResult]):
        """Include related entries in search results."""
        additional_entries = []
        
        for result in results:
            related_entries = await self.get_related_entries(result.entry.entry_id, max_related=2)
            for related_entry in related_entries:
                # Add with lower relevance score
                additional_result = MemorySearchResult(
                    entry=related_entry,
                    relevance_score=result.relevance_score * 0.7,
                    match_reasons=[f"Related to: {result.entry.entry_id}"]
                )
                additional_entries.append(additional_result)
        
        results.extend(additional_entries)
    
    async def _prune_memories(self, 
                            max_age_days: Optional[int] = None,
                            min_relevance_score: Optional[float] = None,
                            target_count: Optional[int] = None) -> int:
        """Internal method to prune memories."""
        async with self._lock:
            pruned_count = 0
            current_time = datetime.now()
            entries_to_remove = []
            
            for entry in self.entries.values():
                should_remove = False
                
                # Age-based pruning
                if max_age_days:
                    age_days = (current_time - entry.created_at).days
                    if age_days > max_age_days and entry.importance != MemoryImportance.CRITICAL:
                        should_remove = True
                
                # Relevance-based pruning
                if min_relevance_score:
                    relevance = entry.calculate_relevance_score(current_time)
                    if relevance < min_relevance_score and entry.importance != MemoryImportance.CRITICAL:
                        should_remove = True
                
                if should_remove:
                    entries_to_remove.append(entry.entry_id)
            
            # Target count pruning (remove lowest relevance entries)
            if target_count and len(self.entries) > target_count:
                # Calculate relevance for all entries
                entry_relevance = []
                for entry in self.entries.values():
                    if entry.importance != MemoryImportance.CRITICAL:  # Never prune critical entries
                        relevance = entry.calculate_relevance_score(current_time)
                        entry_relevance.append((entry.entry_id, relevance))
                
                # Sort by relevance (lowest first)
                entry_relevance.sort(key=lambda x: x[1])
                
                # Add lowest relevance entries to removal list
                entries_to_prune = len(self.entries) - target_count
                for entry_id, _ in entry_relevance[:entries_to_prune]:
                    if entry_id not in entries_to_remove:
                        entries_to_remove.append(entry_id)
            
            # Remove entries
            for entry_id in entries_to_remove:
                if await self.delete_entry(entry_id):
                    pruned_count += 1
            
            if pruned_count > 0:
                logger.info(f"Pruned {pruned_count} memory entries")
            
            return pruned_count
    
    async def _load_memories(self):
        """Load memories from disk."""
        import os
        import json
        
        memory_files = []
        if os.path.exists(self.storage_path):
            memory_files = [f for f in os.listdir(self.storage_path) if f.endswith('.json')]
        
        for filename in memory_files:
            try:
                filepath = os.path.join(self.storage_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                entry = MemoryEntry.model_validate(data)
                self.entries[entry.entry_id] = entry
                
                # Update indexes
                self._update_indexes(entry)
                
                # Update content hash
                content_hash = entry.get_content_hash()
                self.content_hashes[content_hash] = entry.entry_id
                
            except Exception as e:
                logger.error(f"Error loading memory file {filename}: {e}")
    
    async def _save_entry(self, entry: MemoryEntry):
        """Save a memory entry to disk."""
        import os
        import json
        
        filepath = os.path.join(self.storage_path, f"{entry.entry_id}.json")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(entry.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving memory entry {entry.entry_id}: {e}")
    
    async def _delete_entry_file(self, entry_id: str):
        """Delete a memory entry file from disk."""
        import os
        
        filepath = os.path.join(self.storage_path, f"{entry_id}.json")
        
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.error(f"Error deleting memory file {entry_id}: {e}")