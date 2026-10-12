"""The energy gate judged utterances at the wrong moment, and battery warnings
fired on sensor glitches — both proven by the reporter's own output (#45).

His stt_debug run showed Vosk understanding every command perfectly
("cathy debout", peak 635) while the gate dropped ALL of them: Vosk finalizes
a phrase when silence begins, and the gate measured exactly that silence.
His brain journal showed "battery low: 0.0 V" (ADC glitch) and 6.56 V
(servo-load sag) warnings while the pack measured 7.4 V at rest.
"""
import body.nox_behavior_engine as be
import body.nox_voice_loop_v2 as v2
import brain.nox_voice_brain as vb


# ─── energy gate: judge the utterance, not the silence that ends it ───

def test_speech_that_ends_in_silence_passes_the_gate():
    # The reporter's case: loud speech, then the quiet chunks during which
    # Vosk finalizes. The old gate dropped this phrase.
    energy = v2.UtteranceEnergy()
    for rms in [635, 620, 400, 30, 10]:  # speech ... trailing silence
        energy.feed(rms)
    assert energy.harvest() >= v2.ENERGY_WINDOW


def test_a_phantom_from_pure_silence_is_still_dropped():
    energy = v2.UtteranceEnergy()
    for rms in [30, 25, 40, 10]:
        energy.feed(rms)
    assert energy.harvest() < v2.ENERGY_WINDOW


def test_harvest_starts_the_next_utterance_at_zero():
    energy = v2.UtteranceEnergy()
    for rms in [800, 800, 10]:
        energy.feed(rms)
    assert energy.harvest() == 2 and energy.peak == 0
    energy.feed(20)
    assert energy.harvest() == 0  # the loud previous utterance left no residue


def test_the_voice_loop_logs_dropped_phrases():
    # A dropped phrase used to vanish without a trace; now the journal says so.
    src = open("body/nox_voice_loop_v2.py").read()
    assert "Energy gate dropped" in src


def test_stt_debug_uses_the_services_gate_not_a_replica():
    src = open("scripts/stt_debug.py").read()
    assert "UtteranceEnergy()" in src
    # the old replica drifted into a lying message ("peak 635 < 500")
    assert "at the end" not in src


# ─── battery sanity: glitches and load sags are not charge levels ───

def test_a_zero_volt_glitch_is_not_a_battery_level():
    # "battery low: 0.0 V" from the reporter's journal — an unplugged pack or
    # a glitched ADC read, never a charge level.
    assert vb.numeric_battery({"battery_v": 0.0}) is None
    assert vb.numeric_battery({"battery_v": 0.9}) is None
    assert vb.numeric_battery({"battery_v": 1.1}) == 1.1


def test_engine_ignores_a_single_load_sag():
    engine = be.BehaviorEngine(daemon_send_fn=lambda cmd: {"ok": True})
    engine._on_battery(6.56)  # one sag under servo load (reporter's journal)
    assert engine.low_battery_mode is False
    engine._on_battery(7.4)   # recovered — the sag is forgotten
    engine._on_battery(6.56)
    assert engine.low_battery_mode is False


def test_engine_reacts_to_a_sustained_low_battery():
    spoken_texts = []
    engine = be.BehaviorEngine(
        daemon_send_fn=lambda cmd: spoken_texts.append(cmd.get("text"))
        if cmd.get("cmd") == "speak" else {"ok": True})
    engine._on_battery(6.6)
    engine._on_battery(6.6)   # second consecutive reading → act
    assert engine.low_battery_mode is True
    assert len([t for t in spoken_texts if t]) == 1


def test_engine_critical_still_lies_down_when_sustained():
    engine = be.BehaviorEngine(daemon_send_fn=lambda cmd: {"ok": True})
    engine._on_battery(6.0)
    assert engine.critical_battery_mode is False  # one glitch must not drop the dog
    engine._on_battery(6.0)
    assert engine.critical_battery_mode is True and engine.low_battery_mode is True
