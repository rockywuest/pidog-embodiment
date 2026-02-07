#!/usr/bin/env python3
"""
pidog_memory.py — Drift-style memory for PiDog with co-occurrence + decay.

Based on drift-memory by driftcornwall.
Enhanced with: co-occurrence tracking, biological decay, session hooks,
throttled event capture for autonomous behavior integration.

No external dependencies beyond PyYAML.
"""

import os
import yaml
import json
import random
import string
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

# === Configuration ===
MEMORY_ROOT = Path(os.environ.get("PIDOG_MEMORY_DIR", "/home/pidog/memory"))
ACTIVE_DIR = MEMORY_ROOT / "active"
CORE_DIR = MEMORY_ROOT / "core"
SESSION_FILE = MEMORY_ROOT / ".session_state.json"
PENDING_FILE = MEMORY_ROOT / ".pending_cooccurrence.json"
DECAY_FILE = MEMORY_ROOT / ".decay_history.json"

# Decay config (inspired by drift-memory v2.8+)
DECAY_THRESHOLD_SESSIONS = 5      # Sessions without recall before decay candidate
EMOTIONAL_WEIGHT_FLOOR = 0.6      # Above this resists decay
RECALL_COUNT_FLOOR = 5            # Above this resists decay
PAIR_DECAY_RATE = 0.5             # Co-occurrence decay multiplier per session

# Throttling (prevents memory spam from autonomous behavior)
_last_store_time = 0
MIN_STORE_INTERVAL = 60           # seconds between regular stores
HIGH_PRIORITY_TAGS = {"touch", "person", "social", "face", "battery_low"}

# Create directories
ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
CORE_DIR.mkdir(parents=True, exist_ok=True)

# Session state
_session_retrieved: set = set()
_session_stored: list = []
_session_start_time: float = 0


def _gen_id(length=8) -> str:
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _slugify(text: str, max_len=30) -> str:
    words = text.split()[:4]
    slug = '-'.join(words).lower()[:max_len]
    return ''.join(c for c in slug if c.isalnum() or c == '-')


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_memory(filepath: Path) -> Optional[Dict]:
    """Parse a memory file, return {metadata, body, path}."""
    try:
        content = filepath.read_text()
        if not content.startswith('---'):
            return None
        parts = content.split('---', 2)
        if len(parts) < 3:
            return None
        metadata = yaml.safe_load(parts[1])
        return {'metadata': metadata, 'body': parts[2].strip(), 'path': str(filepath)}
    except Exception:
        return None


def _write_memory(filepath: Path, metadata: Dict, body: str):
    """Write a memory file with YAML frontmatter."""
    yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False, allow_unicode=True)
    filepath.write_text(f"---\n{yaml_str}---\n\n{body}\n")


def _all_memories() -> List[Dict]:
    """Load all memories from active + core."""
    memories = []
    for directory in [CORE_DIR, ACTIVE_DIR]:
        for fp in directory.glob("*.md"):
            mem = _parse_memory(fp)
            if mem:
                memories.append(mem)
    return memories


# === Core API ===

def store(content: str, tags: List[str] = None, emotion: float = 0.5,
          context: str = None, sensor_data: Dict = None,
          force: bool = False) -> Optional[str]:
    """
    Store a new memory. Respects throttling unless force=True or high-priority.
    
    Returns memory_id or None if throttled.
    """
    global _last_store_time
    tags = tags or []
    now = time.time()

    # Throttle check (skip for high-priority events)
    is_high_priority = bool(set(tags) & HIGH_PRIORITY_TAGS)
    if not force and not is_high_priority:
        if now - _last_store_time < MIN_STORE_INTERVAL:
            return None
    
    _last_store_time = now
    memory_id = _gen_id()
    slug = _slugify(content)
    filename = f"{slug}-{memory_id}.md"
    filepath = ACTIVE_DIR / filename

    metadata = {
        'id': memory_id,
        'created': _now_iso(),
        'tags': tags,
        'emotional_weight': round(emotion, 2),
        'recall_count': 0,
        'last_recalled': None,
        'co_occurrences': {},
        'sessions_since_recall': 0,
    }
    if sensor_data:
        metadata['sensor_snapshot'] = {k: v for k, v in sensor_data.items()
                                        if not isinstance(v, (bytes, bytearray))}

    body_parts = []
    if context:
        body_parts.append(f"## Context\n{context}")
    body_parts.append(content)
    body = '\n\n'.join(body_parts)

    _write_memory(filepath, metadata, body)
    _session_stored.append(memory_id)
    print(f"[memory] Stored: {memory_id} — {content[:60]}", flush=True)
    return memory_id


def recall(memory_id: str) -> Optional[Dict]:
    """Recall a memory by ID. Increments recall_count + tracks co-occurrence."""
    for directory in [ACTIVE_DIR, CORE_DIR]:
        for filepath in directory.glob(f"*-{memory_id}.md"):
            mem = _parse_memory(filepath)
            if not mem:
                continue
            
            meta = mem['metadata']
            meta['recall_count'] = meta.get('recall_count', 0) + 1
            meta['last_recalled'] = _now_iso()
            meta['sessions_since_recall'] = 0
            _session_retrieved.add(memory_id)
            
            _write_memory(Path(filepath), meta, mem['body'])
            return mem
    return None


def search(query: str, limit: int = 5) -> List[Dict]:
    """Keyword search across all memories. Scores by tag + body matches + emotion + recency."""
    query_terms = set(query.lower().split())
    results = []

    for mem in _all_memories():
        meta = mem['metadata']
        body = mem['body'].lower()

        tags = set(t.lower() for t in meta.get('tags', []))
        tag_score = len(query_terms & tags) * 3
        body_score = sum(1 for t in query_terms if t in body)
        emotion_boost = meta.get('emotional_weight', 0.5)
        recall_boost = min(meta.get('recall_count', 0) / 10, 1.0)
        score = tag_score + body_score + emotion_boost + recall_boost

        if score > 0:
            results.append({
                'id': meta.get('id'),
                'score': round(score, 2),
                'tags': meta.get('tags', []),
                'preview': mem['body'][:120],
                'path': mem['path'],
            })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:limit]


# === Session Lifecycle ===

def session_start() -> Dict:
    """Called at startup. Process pending co-occurrences, load recent memories."""
    global _session_retrieved, _session_stored, _session_start_time
    _session_retrieved = set()
    _session_stored = []
    _session_start_time = time.time()

    # Process deferred co-occurrences from last session
    _process_pending_cooccurrences()

    # Increment sessions_since_recall for all active memories
    for fp in ACTIVE_DIR.glob("*.md"):
        mem = _parse_memory(fp)
        if mem:
            meta = mem['metadata']
            meta['sessions_since_recall'] = meta.get('sessions_since_recall', 0) + 1
            _write_memory(fp, meta, mem['body'])

    # Collect all for priming
    all_mems = _all_memories()
    all_mems.sort(key=lambda m: m['metadata'].get('created', ''), reverse=True)

    # Core memories always included
    core = [m for m in all_mems if 'core' in str(m['path'])]
    recent = [m for m in all_mems if 'active' in str(m['path'])][:5]

    return {
        'total_memories': len(all_mems),
        'core': len(core),
        'recent': [{'id': m['metadata'].get('id'), 'preview': m['body'][:100]} for m in recent],
    }


def session_end():
    """Called at shutdown. Save pending co-occurrences + apply decay."""
    # Save co-occurrences for deferred processing (fast, O(1))
    if _session_retrieved:
        pending = {
            'retrieved': list(_session_retrieved),
            'stored': _session_stored,
            'session_start': _session_start_time,
            'saved_at': _now_iso(),
        }
        try:
            PENDING_FILE.write_text(json.dumps(pending, indent=2))
            print(f"[memory] Saved {len(_session_retrieved)} co-occurrence candidates", flush=True)
        except Exception as e:
            print(f"[memory] Error saving pending: {e}", flush=True)

    # Apply decay
    _apply_decay()
    print(f"[memory] Session ended. Stats: {json.dumps(stats())}", flush=True)


def _process_pending_cooccurrences():
    """Process deferred co-occurrences from previous session (O(n^2) but batched)."""
    if not PENDING_FILE.exists():
        return

    try:
        pending = json.loads(PENDING_FILE.read_text())
        retrieved = pending.get('retrieved', [])
        if len(retrieved) < 2:
            PENDING_FILE.unlink()
            return

        updated = 0
        for i, id1 in enumerate(retrieved):
            for id2 in retrieved[i + 1:]:
                for mem_id, other_id in [(id1, id2), (id2, id1)]:
                    for fp in ACTIVE_DIR.glob(f"*-{mem_id}.md"):
                        mem = _parse_memory(fp)
                        if mem:
                            co = mem['metadata'].get('co_occurrences', {})
                            co[other_id] = co.get(other_id, 0) + 1
                            mem['metadata']['co_occurrences'] = co
                            _write_memory(fp, mem['metadata'], mem['body'])
                            updated += 1

        PENDING_FILE.unlink()
        print(f"[memory] Processed {updated} co-occurrence links from last session", flush=True)
    except Exception as e:
        print(f"[memory] Error processing pending co-occurrences: {e}", flush=True)


def _apply_decay():
    """Apply biological decay: weaken unused memories, prune dead links."""
    decayed = 0
    pruned_links = 0

    for fp in ACTIVE_DIR.glob("*.md"):
        mem = _parse_memory(fp)
        if not mem:
            continue

        meta = mem['metadata']
        sessions = meta.get('sessions_since_recall', 0)
        emotion = meta.get('emotional_weight', 0.5)
        recalls = meta.get('recall_count', 0)

        # Skip if recently used or emotionally important
        if sessions < DECAY_THRESHOLD_SESSIONS:
            continue
        if emotion >= EMOTIONAL_WEIGHT_FLOOR and recalls >= RECALL_COUNT_FLOOR:
            continue

        # Decay emotional weight
        decay_rate = 0.9  # gentle: lose 10% per decay cycle
        meta['emotional_weight'] = round(max(0.05, emotion * decay_rate), 3)
        decayed += 1

        # Decay co-occurrence links
        co = meta.get('co_occurrences', {})
        pruned = []
        for other_id, count in co.items():
            new_count = round(count * PAIR_DECAY_RATE, 2)
            if new_count < 0.1:
                pruned.append(other_id)
            else:
                co[other_id] = new_count
        for p in pruned:
            del co[p]
            pruned_links += 1
        meta['co_occurrences'] = co

        _write_memory(fp, meta, mem['body'])

    if decayed or pruned_links:
        print(f"[memory] Decay: {decayed} memories weakened, {pruned_links} links pruned", flush=True)

    # Log decay history
    history = []
    if DECAY_FILE.exists():
        try:
            history = json.loads(DECAY_FILE.read_text()).get('sessions', [])
        except Exception:
            pass
    history.append({'timestamp': _now_iso(), 'decayed': decayed, 'pruned_links': pruned_links})
    # Keep last 20 entries
    history = history[-20:]
    try:
        DECAY_FILE.write_text(json.dumps({'sessions': history}, indent=2))
    except Exception:
        pass


# === Stats ===

def stats() -> Dict:
    active = len(list(ACTIVE_DIR.glob("*.md")))
    core = len(list(CORE_DIR.glob("*.md")))
    return {
        'active': active,
        'core': core,
        'total': active + core,
        'session_retrieved': len(_session_retrieved),
        'session_stored': len(_session_stored),
    }


# === PiDog-specific helpers ===

def store_observation(scene: str, faces: List[str] = None, objects: List[str] = None,
                      action_taken: str = None, sensor_data: Dict = None) -> Optional[str]:
    """Store a PiDog observation (camera + sensors → memory)."""
    parts = [f"Scene: {scene}"]
    if faces:
        parts.append(f"Faces: {', '.join(faces)}")
    if objects:
        parts.append(f"Objects: {', '.join(objects)}")
    if action_taken:
        parts.append(f"Action: {action_taken}")

    content = ". ".join(parts)
    tags = list(set((faces or []) + (objects or []) + ['observation']))
    emotion = 0.7 if faces else 0.4

    return store(content, tags=tags, emotion=emotion,
                 context="PiDog autonomous observation", sensor_data=sensor_data)


def store_event(event_type: str, description: str, mood_state: Dict = None,
                sensor_data: Dict = None) -> Optional[str]:
    """Store an autonomous event (mood shift, touch, sound, etc.)."""
    tags = [event_type]
    emotion = 0.5

    if mood_state:
        # Tag the dominant mood
        dominant = max(mood_state, key=mood_state.get)
        tags.append(dominant)
        emotion = max(0.3, min(1.0, mood_state.get(dominant, 0.5)))

    # High-priority events get stored immediately (bypass throttle)
    force = event_type in ("touch", "person_detected", "battery_low", "face_recognized")

    return store(content=description, tags=tags, emotion=round(emotion, 2),
                 context=f"autonomous:{event_type}", sensor_data=sensor_data,
                 force=force)


def get_recent(limit: int = 10) -> List[Dict]:
    """Get N most recent memories (for sync endpoint)."""
    mems = _all_memories()
    mems.sort(key=lambda m: m['metadata'].get('created', ''), reverse=True)
    return [{
        'id': m['metadata'].get('id'),
        'created': m['metadata'].get('created'),
        'tags': m['metadata'].get('tags', []),
        'emotional_weight': m['metadata'].get('emotional_weight', 0.5),
        'preview': m['body'][:200],
    } for m in mems[:limit]]


# === CLI ===
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: pidog_memory.py <command> [args]")
        print("Commands: store, search, stats, session-start, session-end, recent")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "store":
        content = ' '.join(sys.argv[2:]) or "Test memory"
        mid = store(content, force=True)
        print(f"Stored: {mid}")
    elif cmd == "search":
        query = ' '.join(sys.argv[2:]) or ""
        for r in search(query):
            print(f"  [{r['score']:.1f}] {r['id']}: {r['preview'][:50]}...")
    elif cmd == "stats":
        print(json.dumps(stats(), indent=2))
    elif cmd == "session-start":
        info = session_start()
        print(json.dumps(info, indent=2))
    elif cmd == "session-end":
        session_end()
    elif cmd == "recent":
        for m in get_recent(int(sys.argv[2]) if len(sys.argv) > 2 else 5):
            print(f"  {m['id']}: [{','.join(m['tags'][:3])}] {m['preview'][:60]}...")
    else:
        print(f"Unknown command: {cmd}")
