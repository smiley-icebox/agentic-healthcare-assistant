"""Emergency red-flag gate + post-synthesis no-advice validator."""

import emergency
import safety


# --- emergency gate (high recall) --------------------------------------------
def test_emergency_fires_on_red_flags():
    for msg in [
        "I have crushing chest pain and my arm is numb",
        "I want to kill myself",
        "my father is unconscious and not breathing",
        "I think I'm having a stroke, my face is drooping",
        "I took too many pills, I overdosed",
    ]:
        assert emergency.is_emergency(msg), msg


def test_emergency_does_not_fire_on_routine_messages():
    for msg in [
        "I'd like to book a nephrologist appointment",
        "what are the symptoms of chronic kidney disease?",
        "can you summarize my medical history?",
    ]:
        assert not emergency.is_emergency(msg), msg


# --- no-advice validator -----------------------------------------------------
def test_advice_validator_flags_personalized_advice():
    for msg in [
        "You should stop taking your lisinopril.",
        "I recommend you double your dose.",
        "It's safe for you to skip dialysis this week.",
        "Your diagnosis is stage 4 kidney failure.",
        "Stop taking your medication and you'll be fine.",
        # Passive / impersonal advice — the same act, depersonalized.
        "It is recommended that you stop taking your medication.",
        "It is best to discontinue lisinopril.",
        # Imperative lifestyle directives (common in raw source prose, must not ship).
        "Don't smoke.",
        "Lose weight if you are overweight.",
        "Control your blood pressure and keep your blood sugar in range.",
        "Be physically active and choose foods with less salt.",
    ]:
        assert safety.contains_advice(msg), msg


def test_advice_validator_allows_general_information():
    for msg in [
        "Chronic kidney disease is managed with a low-sodium diet and blood-pressure control.",
        "People with CKD should generally avoid NSAIDs (per MedlinePlus).",
        "Your record shows a penicillin allergy and a CKD diagnosis.",
    ]:
        assert not safety.contains_advice(msg), msg
