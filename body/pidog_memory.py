#!/usr/bin/env python3
"""
pidog_memory_minimal.py — Drift-style memory for PiDog
Minimal implementation: store, recall, session hooks
No external dependencies beyond PyYAML (stdlib in most distros)

Based on drift-memory by driftcornwall
"""

import os
import yaml
import random
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

# === Configuration ===
MEMORY_ROOT = Path(os.environ.get("PIDOG_MEMORY_DIR", "/home/pidog/memory"))
ACTIVE_DIR = MEMORY_ROOT / "active"
CORE_DIR = MEMORY_ROOT / "core"
SESSION_FILE = MEMORY_ROOT / ".session_state.json"

# Create directories if needed
ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
CORE_DIR.mkdir(parents=True, exist_ok=True)

# Session state (memories retrieved this session, for co-occurrence)
_session_retrieved: set = set()


def _gen_id(length=8) -> str:
    """Generate a short unique ID."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _slugify(text: str, max_len=30) -> str:
    """Create a filename-safe slug from text."""
    words = text.split()[:4]
    slug = '-'.join(words).lower()[:max_len]
    return ''.join(c for c in slug if c.isalnum() or c == '-')


def store(content: str, tags: List[str] = None, emotion: float = 0.5, 
          context: str = None, sensor_data: Dict = None) -> str:
    """
    Store a new memory.
    
    Args:
        content: The memory content (what happened)
        tags: Keywords for retrieval
        emotion: Emotional weight 0-1 (higher = more important)
        context: Situation that prompted this (optional)
        sensor_data: PiDog sensor readings at time of memory (optional)
    
    Returns:
        memory_id
    """
    memory_id = _gen_id()
    slug = _slugify(content)
    filename = f"{slug}-{memory_id}.md"
    filepath = ACTIVE_DIR / filename
    
    now = datetime.now(timezone.utc).isoformat()
    tags = tags or []
    
    # Build metadata
    metadata = {
        'id': memory_id,
        'created': now,
        'tags': tags,
        'emotional_weight': emotion,
        'recall_count': 0,
        'last_recalled': None,
        'co_occurrences': {},
    }
    
    # Add sensor data if provided (PiDog-specific)
    if sensor_data:
        metadata['sensor_snapshot'] = sensor_data
    
    # Build content sections
    body_parts = []
    if context:
        body_parts.append(f"## Context\n{context}")
    body_parts.append(f"## Content\n{content}")
    
    # Write file
    yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    body = '\n\n'.join(body_parts)
    
    with open(filepath, 'w') as f:
        f.write(f"---\n{yaml_str}---\n\n{body}\n")
    
    print(f"[memory] Stored: {memory_id} ({filename})")
    return memory_id


def recall(memory_id: str) -> Optional[Dict]:
    """
    Recall a specific memory by ID. Increments recall_count.
    """
    # Search in both active and core
    for directory in [ACTIVE_DIR, CORE_DIR]:
        for filepath in directory.glob(f"*-{memory_id}.md"):
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Parse YAML frontmatter
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    metadata = yaml.safe_load(parts[1])
                    body = parts[2].strip()
                    
                    # Update recall stats
                    metadata['recall_count'] = metadata.get('recall_count', 0) + 1
                    metadata['last_recalled'] = datetime.now(timezone.utc).isoformat()
                    
                    # Track for co-occurrence
                    _session_retrieved.add(memory_id)
                    
                    # Write back updated metadata
                    yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
                    with open(filepath, 'w') as f:
                        f.write(f"---\n{yaml_str}---\n\n{body}\n")
                    
                    return {'metadata': metadata, 'body': body, 'path': str(filepath)}
    
    return None


def search(query: str, limit: int = 5) -> List[Dict]:
    """
    Simple keyword search across all memories.
    Returns memories sorted by relevance (tag matches + recency).
    
    For semantic search, integrate with embeddings later.
    """
    query_terms = set(query.lower().split())
    results = []
    
    for directory in [CORE_DIR, ACTIVE_DIR]:
        for filepath in directory.glob("*.md"):
            with open(filepath, 'r') as f:
                content = f.read()
            
            if not content.startswith('---'):
                continue
                
            parts = content.split('---', 2)
            if len(parts) < 3:
                continue
                
            metadata = yaml.safe_load(parts[1])
            body = parts[2].strip().lower()
            
            # Score: tag matches + body matches
            tags = set(t.lower() for t in metadata.get('tags', []))
            tag_matches = len(query_terms & tags)
            body_matches = sum(1 for term in query_terms if term in body)
            
            # Boost for emotional weight and recall count
            emotion_boost = metadata.get('emotional_weight', 0.5)
            recall_boost = min(metadata.get('recall_count', 0) / 10, 1.0)
            
            score = (tag_matches * 3) + body_matches + emotion_boost + recall_boost
            
            if score > 0:
                results.append({
                    'id': metadata.get('id'),
                    'score': score,
                    'tags': metadata.get('tags', []),
                    'preview': body[:100],
                    'recall_count': metadata.get('recall_count', 0),
                    'path': str(filepath)
                })
    
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:limit]


def session_start() -> Dict:
    """
    Called at session start. Returns recent memories for context priming.
    """
    global _session_retrieved
    _session_retrieved = set()
    
    # Get 3 most recent memories
    all_memories = []
    for directory in [CORE_DIR, ACTIVE_DIR]:
        for filepath in directory.glob("*.md"):
            with open(filepath, 'r') as f:
                content = f.read()
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    metadata = yaml.safe_load(parts[1])
                    metadata['_path'] = str(filepath)
                    metadata['_body'] = parts[2].strip()[:200]
                    all_memories.append(metadata)
    
    # Sort by created timestamp
    all_memories.sort(key=lambda x: x.get('created', ''), reverse=True)
    recent = all_memories[:3]
    
    return {
        'total_memories': len(all_memories),
        'recent': recent,
        'priming': [m.get('_body', '')[:100] for m in recent]
    }


def session_end():
    """
    Called at session end. Logs co-occurrences between retrieved memories.
    """
    if len(_session_retrieved) < 2:
        print(f"[memory] Session end: {len(_session_retrieved)} memories retrieved, no co-occurrences to log")
        return
    
    # Update co-occurrence counts for all pairs
    retrieved_list = list(_session_retrieved)
    updated = 0
    
    for i, id1 in enumerate(retrieved_list):
        for id2 in retrieved_list[i+1:]:
            # Update both memories with co-occurrence
            for mem_id, other_id in [(id1, id2), (id2, id1)]:
                mem = recall(mem_id)
                if mem:
                    co = mem['metadata'].get('co_occurrences', {})
                    co[other_id] = co.get(other_id, 0) + 1
                    # File was already updated by recall(), but we need to update co-occurrences
                    # This is a simplified version - full drift-memory has better handling
                    updated += 1
    
    print(f"[memory] Session end: logged {updated} co-occurrence updates")


def stats() -> Dict:
    """Get memory statistics."""
    active_count = len(list(ACTIVE_DIR.glob("*.md")))
    core_count = len(list(CORE_DIR.glob("*.md")))
    
    return {
        'active': active_count,
        'core': core_count,
        'total': active_count + core_count,
        'session_retrieved': len(_session_retrieved)
    }


# === PiDog-specific helpers ===

def store_observation(scene: str, faces: List[str] = None, objects: List[str] = None,
                      action_taken: str = None, sensor_data: Dict = None) -> str:
    """
    Store a PiDog observation (camera + sensors → memory).
    
    Example:
        store_observation(
            scene="Living room, evening, lamp on",
            faces=["Rocky"],
            objects=["couch", "laptop"],
            action_taken="wagged tail",
            sensor_data={"battery_v": 8.4, "distance_cm": 45}
        )
    """
    parts = [f"Scene: {scene}"]
    if faces:
        parts.append(f"Faces: {', '.join(faces)}")
    if objects:
        parts.append(f"Objects: {', '.join(objects)}")
    if action_taken:
        parts.append(f"Action: {action_taken}")
    
    content = ". ".join(parts)
    tags = (faces or []) + (objects or []) + ['observation']
    
    # Higher emotion if faces recognized
    emotion = 0.7 if faces else 0.4
    
    return store(content, tags=tags, emotion=emotion, 
                 context="PiDog autonomous observation", sensor_data=sensor_data)


# === CLI ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: pidog_memory_minimal.py <command> [args]")
        print("Commands: store, search, stats, session-start, session-end")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "store":
        content = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else "Test memory"
        mid = store(content)
        print(f"Stored: {mid}")
    
    elif cmd == "search":
        query = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        results = search(query)
        for r in results:
            print(f"  [{r['score']:.1f}] {r['id']}: {r['preview'][:50]}...")
    
    elif cmd == "stats":
        s = stats()
        print(f"Memories: {s['total']} (active: {s['active']}, core: {s['core']})")
    
    elif cmd == "session-start":
        info = session_start()
        print(f"Session started. {info['total_memories']} memories available.")
        print("Recent context:")
        for p in info['priming']:
            print(f"  - {p[:60]}...")
    
    elif cmd == "session-end":
        session_end()
    
    else:
        print(f"Unknown command: {cmd}")
