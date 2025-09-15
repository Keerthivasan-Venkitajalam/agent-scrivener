"""
Unit tests for MemoryStore.
"""

import pytest
import tempfile
import shutil
import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from agent_scrivener.memory.memory_store import (
    MemoryStore, MemoryEntry, MemoryType, MemoryImportance,
    MemoryQuery, MemorySearchResult
)
from agent_scrivener.models.core import Insight, Source, SourceType


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def memory_store(temp_storage):
    """Create memory store with temporary storage."""
    return MemoryStore(storage_path=temp_storage, max_entries=100)


@pytest.fixture
def sample_insight():
    """Create a sample insight."""
    source = Source(
        url="https://example.com/article",
        title="Test Article",
        source_type=SourceType.WEB
    )
    
    return Insight(
        topic="Machine Learning",
        summary="Machine learning is transforming research automation",
        supporting_evidence=["Evidence 1", "Evidence 2"],
        confidence_score=0.8,
        related_sources=[source],
        tags=["ml", "automation", "research"]
    )


@pytest.fixture
def sample_memory_entry():
    """Create a sample memory entry."""
    return MemoryEntry(
        memory_type=MemoryType.INSIGHT,
        content="Test insight content",
        tags=["test", "sample"],
        importance=MemoryImportance.MEDIUM,
        confidence_score=0.7,
        source_references=["https://example.com"]
    )


class TestMemoryEntry:
    """Test cases for MemoryEntry."""
    
    def test_update_access(self, sample_memory_entry):
        """Test updating access statistics."""
        original_access_count = sample_memory_entry.access_count
        original_last_accessed = sample_memory_entry.last_accessed
        
        sample_memory_entry.update_access()
        
        assert sample_memory_entry.access_count == original_access_count + 1
        assert sample_memory_entry.last_accessed > original_last_accessed
    
    def test_calculate_relevance_score(self, sample_memory_entry):
        """Test relevance score calculation."""
        current_time = datetime.now()
        
        # Test with fresh entry
        score = sample_memory_entry.calculate_relevance_score(current_time)
        assert 0.0 <= score <= 1.0
        
        # Test with old entry
        sample_memory_entry.created_at = current_time - timedelta(days=60)
        old_score = sample_memory_entry.calculate_relevance_score(current_time)
        assert old_score < score  # Should be lower for older entries
        
        # Test with high importance
        sample_memory_entry.importance = MemoryImportance.CRITICAL
        critical_score = sample_memory_entry.calculate_relevance_score(current_time)
        assert critical_score > old_score  # Should be higher for critical entries
    
    def test_get_content_hash(self, sample_memory_entry):
        """Test content hash generation."""
        hash1 = sample_memory_entry.get_content_hash()
        hash2 = sample_memory_entry.get_content_hash()
        
        # Same entry should produce same hash
        assert hash1 == hash2
        
        # Different content should produce different hash
        sample_memory_entry.content = "Different content"
        hash3 = sample_memory_entry.get_content_hash()
        assert hash1 != hash3


class TestMemoryStore:
    """Test cases for MemoryStore."""
    
    async def test_initialize(self, memory_store, temp_storage):
        """Test memory store initialization."""
        await memory_store.initialize()
        
        # Verify storage directory exists
        assert os.path.exists(temp_storage)
        assert os.path.isdir(temp_storage)
        
        # Verify initial state
        assert len(memory_store.entries) == 0
        assert len(memory_store.content_hashes) == 0
        assert len(memory_store.tag_index) == 0
        assert len(memory_store.type_index) == 0
    
    async def test_store_insight(self, memory_store, sample_insight):
        """Test storing an insight."""
        await memory_store.initialize()
        
        entry_id = await memory_store.store_insight(sample_insight, MemoryImportance.HIGH)
        
        assert entry_id is not None
        assert entry_id in memory_store.entries
        
        entry = memory_store.entries[entry_id]
        assert entry.memory_type == MemoryType.INSIGHT
        assert entry.content == sample_insight.summary
        assert entry.importance == MemoryImportance.HIGH
        assert entry.confidence_score == sample_insight.confidence_score
        assert set(entry.tags) == set(sample_insight.tags)
    
    async def test_store_source(self, memory_store):
        """Test storing a source."""
        await memory_store.initialize()
        
        source = Source(
            url="https://example.com/test",
            title="Test Source",
            author="Test Author",
            source_type=SourceType.ACADEMIC
        )
        
        entry_id = await memory_store.store_source(source, MemoryImportance.MEDIUM)
        
        assert entry_id is not None
        assert entry_id in memory_store.entries
        
        entry = memory_store.entries[entry_id]
        assert entry.memory_type == MemoryType.SOURCE
        assert source.title in entry.content
        assert entry.importance == MemoryImportance.MEDIUM
        assert "academic" in entry.tags
    
    async def test_store_fact(self, memory_store):
        """Test storing a fact."""
        await memory_store.initialize()
        
        fact = "The sky is blue"
        sources = ["https://example.com/sky"]
        confidence = 0.9
        tags = ["color", "sky"]
        
        entry_id = await memory_store.store_fact(
            fact, sources, confidence, tags, MemoryImportance.LOW
        )
        
        assert entry_id is not None
        assert entry_id in memory_store.entries
        
        entry = memory_store.entries[entry_id]
        assert entry.memory_type == MemoryType.FACT
        assert entry.content == fact
        assert entry.confidence_score == confidence
        assert entry.tags == tags
        assert entry.source_references == sources
        assert entry.importance == MemoryImportance.LOW
    
    async def test_add_entry_duplicate(self, memory_store, sample_memory_entry):
        """Test adding duplicate entry."""
        await memory_store.initialize()
        
        # Add first entry
        entry_id1 = await memory_store.add_entry(sample_memory_entry)
        
        # Try to add duplicate
        duplicate_entry = MemoryEntry(
            memory_type=sample_memory_entry.memory_type,
            content=sample_memory_entry.content,
            tags=sample_memory_entry.tags,
            importance=MemoryImportance.LOW,  # Different importance
            confidence_score=0.5  # Different confidence
        )
        
        entry_id2 = await memory_store.add_entry(duplicate_entry)
        
        # Should return same entry ID
        assert entry_id1 == entry_id2
        
        # Should have updated the existing entry
        entry = memory_store.entries[entry_id1]
        assert entry.importance == MemoryImportance.MEDIUM  # Should keep higher importance
        assert entry.confidence_score == 0.7  # Should keep higher confidence
    
    async def test_get_entry(self, memory_store, sample_memory_entry):
        """Test getting an entry."""
        await memory_store.initialize()
        
        entry_id = await memory_store.add_entry(sample_memory_entry)
        original_access_count = sample_memory_entry.access_count
        
        retrieved_entry = await memory_store.get_entry(entry_id)
        
        assert retrieved_entry is not None
        assert retrieved_entry.entry_id == entry_id
        assert retrieved_entry.access_count == original_access_count + 1
    
    async def test_get_nonexistent_entry(self, memory_store):
        """Test getting non-existent entry."""
        await memory_store.initialize()
        
        entry = await memory_store.get_entry("nonexistent")
        
        assert entry is None
    
    async def test_search_memories_basic(self, memory_store):
        """Test basic memory search."""
        await memory_store.initialize()
        
        # Add test entries
        entry1 = MemoryEntry(
            memory_type=MemoryType.INSIGHT,
            content="Machine learning algorithms",
            tags=["ml", "algorithms"],
            importance=MemoryImportance.HIGH,
            confidence_score=0.8
        )
        
        entry2 = MemoryEntry(
            memory_type=MemoryType.FACT,
            content="Deep learning is a subset of machine learning",
            tags=["ml", "deep-learning"],
            importance=MemoryImportance.MEDIUM,
            confidence_score=0.9
        )
        
        await memory_store.add_entry(entry1)
        await memory_store.add_entry(entry2)
        
        # Search for machine learning
        query = MemoryQuery(query_text="machine learning")
        results = await memory_store.search_memories(query)
        
        assert len(results) == 2
        assert all(isinstance(r, MemorySearchResult) for r in results)
        assert all(r.relevance_score > 0 for r in results)
    
    async def test_search_memories_with_filters(self, memory_store):
        """Test memory search with filters."""
        await memory_store.initialize()
        
        # Add test entries
        entry1 = MemoryEntry(
            memory_type=MemoryType.INSIGHT,
            content="Test insight",
            tags=["test"],
            importance=MemoryImportance.HIGH,
            confidence_score=0.8
        )
        
        entry2 = MemoryEntry(
            memory_type=MemoryType.FACT,
            content="Test fact",
            tags=["test"],
            importance=MemoryImportance.LOW,
            confidence_score=0.5
        )
        
        await memory_store.add_entry(entry1)
        await memory_store.add_entry(entry2)
        
        # Search with type filter
        query = MemoryQuery(
            query_text="test",
            memory_types=[MemoryType.INSIGHT]
        )
        results = await memory_store.search_memories(query)
        
        assert len(results) == 1
        assert results[0].entry.memory_type == MemoryType.INSIGHT
        
        # Search with confidence filter
        query = MemoryQuery(
            query_text="test",
            min_confidence=0.7
        )
        results = await memory_store.search_memories(query)
        
        assert len(results) == 1
        assert results[0].entry.confidence_score >= 0.7
        
        # Search with importance filter
        query = MemoryQuery(
            query_text="test",
            min_importance=MemoryImportance.HIGH
        )
        results = await memory_store.search_memories(query)
        
        assert len(results) == 1
        assert results[0].entry.importance == MemoryImportance.HIGH
    
    async def test_search_memories_with_tags(self, memory_store):
        """Test memory search with tag filters."""
        await memory_store.initialize()
        
        # Add entries with different tags
        entry1 = MemoryEntry(
            memory_type=MemoryType.INSIGHT,
            content="ML insight",
            tags=["ml", "ai"],
            importance=MemoryImportance.MEDIUM,
            confidence_score=0.8
        )
        
        entry2 = MemoryEntry(
            memory_type=MemoryType.FACT,
            content="Research fact",
            tags=["research", "data"],
            importance=MemoryImportance.MEDIUM,
            confidence_score=0.7
        )
        
        await memory_store.add_entry(entry1)
        await memory_store.add_entry(entry2)
        
        # Search with tag filter
        query = MemoryQuery(
            query_text="test",
            tags=["ml"]
        )
        results = await memory_store.search_memories(query)
        
        assert len(results) == 1
        assert "ml" in results[0].entry.tags
    
    async def test_search_memories_with_age_filter(self, memory_store):
        """Test memory search with age filter."""
        await memory_store.initialize()
        
        # Add old entry
        old_entry = MemoryEntry(
            memory_type=MemoryType.FACT,
            content="Old fact",
            tags=["old"],
            importance=MemoryImportance.MEDIUM,
            confidence_score=0.7
        )
        old_entry.created_at = datetime.now() - timedelta(days=10)
        
        # Add new entry
        new_entry = MemoryEntry(
            memory_type=MemoryType.FACT,
            content="New fact",
            tags=["new"],
            importance=MemoryImportance.MEDIUM,
            confidence_score=0.7
        )
        
        await memory_store.add_entry(old_entry)
        await memory_store.add_entry(new_entry)
        
        # Search with age filter
        query = MemoryQuery(
            query_text="fact",
            max_age_days=5
        )
        results = await memory_store.search_memories(query)
        
        assert len(results) == 1
        assert "new" in results[0].entry.tags
    
    async def test_get_related_entries(self, memory_store):
        """Test getting related entries."""
        await memory_store.initialize()
        
        # Add entries with similar tags
        entry1 = MemoryEntry(
            memory_type=MemoryType.INSIGHT,
            content="ML insight",
            tags=["ml", "ai"],
            importance=MemoryImportance.MEDIUM,
            confidence_score=0.8
        )
        
        entry2 = MemoryEntry(
            memory_type=MemoryType.FACT,
            content="AI fact",
            tags=["ai", "research"],
            importance=MemoryImportance.MEDIUM,
            confidence_score=0.7
        )
        
        entry3 = MemoryEntry(
            memory_type=MemoryType.FACT,
            content="Unrelated fact",
            tags=["other"],
            importance=MemoryImportance.MEDIUM,
            confidence_score=0.6
        )
        
        entry1_id = await memory_store.add_entry(entry1)
        await memory_store.add_entry(entry2)
        await memory_store.add_entry(entry3)
        
        # Get related entries
        related = await memory_store.get_related_entries(entry1_id)
        
        assert len(related) >= 1
        # Should find entry2 due to shared "ai" tag
        related_tags = [tag for entry in related for tag in entry.tags]
        assert "ai" in related_tags
    
    async def test_add_relationship(self, memory_store):
        """Test adding relationships between entries."""
        await memory_store.initialize()
        
        entry1 = MemoryEntry(
            memory_type=MemoryType.INSIGHT,
            content="First insight",
            tags=["test"],
            importance=MemoryImportance.MEDIUM,
            confidence_score=0.8
        )
        
        entry2 = MemoryEntry(
            memory_type=MemoryType.FACT,
            content="Related fact",
            tags=["test"],
            importance=MemoryImportance.MEDIUM,
            confidence_score=0.7
        )
        
        entry1_id = await memory_store.add_entry(entry1)
        entry2_id = await memory_store.add_entry(entry2)
        
        # Add relationship
        result = await memory_store.add_relationship(entry1_id, entry2_id)
        
        assert result is True
        
        # Verify bidirectional relationship
        updated_entry1 = await memory_store.get_entry(entry1_id)
        updated_entry2 = await memory_store.get_entry(entry2_id)
        
        assert entry2_id in updated_entry1.related_entries
        assert entry1_id in updated_entry2.related_entries
    
    async def test_add_relationship_nonexistent(self, memory_store):
        """Test adding relationship with non-existent entry."""
        await memory_store.initialize()
        
        entry = MemoryEntry(
            memory_type=MemoryType.FACT,
            content="Test fact",
            tags=["test"],
            importance=MemoryImportance.MEDIUM,
            confidence_score=0.7
        )
        
        entry_id = await memory_store.add_entry(entry)
        
        # Try to add relationship with non-existent entry
        result = await memory_store.add_relationship(entry_id, "nonexistent")
        
        assert result is False
    
    async def test_delete_entry(self, memory_store, sample_memory_entry):
        """Test deleting an entry."""
        await memory_store.initialize()
        
        entry_id = await memory_store.add_entry(sample_memory_entry)
        
        # Verify entry exists
        assert entry_id in memory_store.entries
        
        # Delete entry
        result = await memory_store.delete_entry(entry_id)
        
        assert result is True
        assert entry_id not in memory_store.entries
        
        # Verify indexes were updated
        for tag in sample_memory_entry.tags:
            if tag in memory_store.tag_index:
                assert entry_id not in memory_store.tag_index[tag]
    
    async def test_delete_nonexistent_entry(self, memory_store):
        """Test deleting non-existent entry."""
        await memory_store.initialize()
        
        result = await memory_store.delete_entry("nonexistent")
        
        assert result is False
    
    async def test_prune_memories_by_age(self, memory_store):
        """Test pruning memories by age."""
        await memory_store.initialize()
        
        # Add old entry
        old_entry = MemoryEntry(
            memory_type=MemoryType.FACT,
            content="Old fact",
            tags=["old"],
            importance=MemoryImportance.MEDIUM,
            confidence_score=0.7
        )
        old_entry.created_at = datetime.now() - timedelta(days=40)
        
        # Add new entry
        new_entry = MemoryEntry(
            memory_type=MemoryType.FACT,
            content="New fact",
            tags=["new"],
            importance=MemoryImportance.MEDIUM,
            confidence_score=0.7
        )
        
        await memory_store.add_entry(old_entry)
        await memory_store.add_entry(new_entry)
        
        # Prune entries older than 30 days
        pruned_count = await memory_store.prune_memories(max_age_days=30)
        
        assert pruned_count == 1
        assert len(memory_store.entries) == 1
        
        # Verify the new entry remains
        remaining_entry = list(memory_store.entries.values())[0]
        assert "new" in remaining_entry.tags
    
    async def test_prune_memories_by_relevance(self, memory_store):
        """Test pruning memories by relevance."""
        await memory_store.initialize()
        
        # Add low relevance entry
        low_entry = MemoryEntry(
            memory_type=MemoryType.FACT,
            content="Low relevance fact",
            tags=["low"],
            importance=MemoryImportance.LOW,
            confidence_score=0.3
        )
        low_entry.created_at = datetime.now() - timedelta(days=30)
        low_entry.last_accessed = datetime.now() - timedelta(days=20)
        
        # Add high relevance entry
        high_entry = MemoryEntry(
            memory_type=MemoryType.INSIGHT,
            content="High relevance insight",
            tags=["high"],
            importance=MemoryImportance.HIGH,
            confidence_score=0.9
        )
        
        await memory_store.add_entry(low_entry)
        await memory_store.add_entry(high_entry)
        
        # Prune entries with low relevance
        pruned_count = await memory_store.prune_memories(min_relevance_score=0.5)
        
        assert pruned_count == 1
        assert len(memory_store.entries) == 1
        
        # Verify the high relevance entry remains
        remaining_entry = list(memory_store.entries.values())[0]
        assert "high" in remaining_entry.tags
    
    async def test_prune_memories_by_target_count(self, memory_store):
        """Test pruning memories to target count."""
        await memory_store.initialize()
        
        # Add multiple entries
        for i in range(5):
            entry = MemoryEntry(
                memory_type=MemoryType.FACT,
                content=f"Fact {i}",
                tags=[f"tag{i}"],
                importance=MemoryImportance.MEDIUM,
                confidence_score=0.5 + (i * 0.1)  # Varying confidence
            )
            await memory_store.add_entry(entry)
        
        # Prune to target count of 3
        pruned_count = await memory_store.prune_memories(target_count=3)
        
        assert pruned_count == 2
        assert len(memory_store.entries) == 3
        
        # Verify higher confidence entries remain
        remaining_confidences = [entry.confidence_score for entry in memory_store.entries.values()]
        assert all(conf >= 0.7 for conf in remaining_confidences)
    
    async def test_prune_memories_preserve_critical(self, memory_store):
        """Test that critical entries are preserved during pruning."""
        await memory_store.initialize()
        
        # Add critical entry
        critical_entry = MemoryEntry(
            memory_type=MemoryType.INSIGHT,
            content="Critical insight",
            tags=["critical"],
            importance=MemoryImportance.CRITICAL,
            confidence_score=0.5  # Low confidence but critical
        )
        critical_entry.created_at = datetime.now() - timedelta(days=100)  # Very old
        
        # Add regular entry
        regular_entry = MemoryEntry(
            memory_type=MemoryType.FACT,
            content="Regular fact",
            tags=["regular"],
            importance=MemoryImportance.MEDIUM,
            confidence_score=0.9
        )
        
        await memory_store.add_entry(critical_entry)
        await memory_store.add_entry(regular_entry)
        
        # Prune with aggressive settings
        pruned_count = await memory_store.prune_memories(
            max_age_days=30,
            min_relevance_score=0.8,
            target_count=1
        )
        
        # Critical entry should be preserved
        assert len(memory_store.entries) >= 1
        critical_preserved = any(
            entry.importance == MemoryImportance.CRITICAL 
            for entry in memory_store.entries.values()
        )
        assert critical_preserved
    
    async def test_get_statistics(self, memory_store):
        """Test getting memory store statistics."""
        await memory_store.initialize()
        
        # Add various entries
        entries_data = [
            (MemoryType.INSIGHT, MemoryImportance.HIGH, 0.8),
            (MemoryType.FACT, MemoryImportance.MEDIUM, 0.7),
            (MemoryType.SOURCE, MemoryImportance.LOW, 0.6),
        ]
        
        for memory_type, importance, confidence in entries_data:
            entry = MemoryEntry(
                memory_type=memory_type,
                content=f"Test {memory_type.value}",
                tags=["test"],
                importance=importance,
                confidence_score=confidence
            )
            await memory_store.add_entry(entry)
        
        # Get statistics
        stats = await memory_store.get_statistics()
        
        assert stats.total_entries == 3
        assert stats.entries_by_type["insight"] == 1
        assert stats.entries_by_type["fact"] == 1
        assert stats.entries_by_type["source"] == 1
        assert stats.entries_by_importance["high"] == 1
        assert stats.entries_by_importance["medium"] == 1
        assert stats.entries_by_importance["low"] == 1
        assert stats.average_confidence == 0.7
        assert stats.oldest_entry_age_days is not None
        assert stats.most_accessed_entry_id is not None
    
    async def test_max_entries_limit(self, temp_storage):
        """Test that memory store respects max entries limit."""
        # Create store with small limit
        memory_store = MemoryStore(storage_path=temp_storage, max_entries=3)
        await memory_store.initialize()
        
        # Add entries beyond limit
        for i in range(5):
            entry = MemoryEntry(
                memory_type=MemoryType.FACT,
                content=f"Fact {i}",
                tags=[f"tag{i}"],
                importance=MemoryImportance.MEDIUM,
                confidence_score=0.5
            )
            await memory_store.add_entry(entry)
        
        # Should not exceed max entries (with some pruning buffer)
        assert len(memory_store.entries) <= 3
    
    @patch('agent_scrivener.memory.memory_store.os.path.exists')
    @patch('agent_scrivener.memory.memory_store.os.listdir')
    async def test_load_memories_error_handling(self, mock_listdir, mock_exists, memory_store):
        """Test error handling during memory loading."""
        mock_exists.return_value = True
        mock_listdir.return_value = ['corrupted.json', 'valid.json']
        
        # Mock file reading to simulate corruption
        with patch('builtins.open', side_effect=[
            # First file - corrupted
            Exception("File read error"),
            # Second file - valid
            MagicMock()
        ]):
            await memory_store.initialize()
        
        # Should handle errors gracefully
        assert len(memory_store.entries) == 0  # No entries loaded due to mocking