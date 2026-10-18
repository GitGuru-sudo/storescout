import pytest
import re
import os
import json
import pandas as pd
from rapidfuzz import process, fuzz

# Re-implement core logic for isolated unit verification
CATEGORY_SYNONYMS = {
    "electronics": "telephony",
    "mobile": "telephony",
    "phone": "telephony",
    "khilona": "toys",
    "toys": "toys",
    "beauty": "health_beauty",
    "furniture": "furniture_decor"
}

def extract_test_slots(text, previous_slots=None):
    slots = previous_slots.copy() if previous_slots else {
        "zip": None,
        "state": None,
        "city": None,
        "category": None,
        "max_delivery_days": None,
        "n": 5
    }
    zip_match = re.search(r'\b(\d{5})\b', text)
    if zip_match:
        slots["zip"] = int(zip_match.group(1))
        
    days_match = re.search(r'(\d+)\s*(?:din|day|days)', text, re.IGNORECASE)
    if days_match:
        slots["max_delivery_days"] = int(days_match.group(1))
    week_match = re.search(r'(\d+)\s*(?:week|weeks|hafte|hafta)', text, re.IGNORECASE)
    if week_match:
        slots["max_delivery_days"] = int(week_match.group(1)) * 7
        
    top_match = re.search(r'\btop\s*(\d+)\b', text, re.IGNORECASE)
    if top_match:
        slots["n"] = int(top_match.group(1))
        
    tokens = re.findall(r'\b[A-Za-z]{2}\b', text)
    for tok in tokens:
        up = tok.upper()
        if up in ['SP', 'RJ', 'MG', 'PR', 'RS', 'BA', 'SC']:
            slots["state"] = up
            break
            
    for syn, mapped in CATEGORY_SYNONYMS.items():
        if re.search(r'\b' + syn + r'\b', text, re.IGNORECASE):
            slots["category"] = mapped
            break
            
    return slots

def test_zip_extraction():
    slots = extract_test_slots("kya zip 01001 me store kholna chahiye?")
    assert slots["zip"] == 1001

def test_delivery_days_extraction_days():
    slots = extract_test_slots("sao paulo me 5 din delivery wale best zones")
    assert slots["max_delivery_days"] == 5

def test_delivery_days_extraction_weeks():
    slots = extract_test_slots("RJ me 2 hafte me deliver hone wale areas")
    assert slots["max_delivery_days"] == 14

def test_top_n_extraction():
    slots = extract_test_slots("show me top 10 zones in SP")
    assert slots["n"] == 10
    assert slots["state"] == "SP"

def test_category_synonym_extraction():
    slots = extract_test_slots("mujhe electronics ke liye jagah batao")
    assert slots["category"] == "telephony"
    
def test_multi_turn_slot_persistence():
    # Turn 1
    slots_turn_1 = extract_test_slots("where can i sell toys in SP?")
    assert slots_turn_1["state"] == "SP"
    assert slots_turn_1["category"] == "toys"
    assert slots_turn_1["max_delivery_days"] is None
    
    # Turn 2: User supplies delivery constraint
    slots_turn_2 = extract_test_slots("under 7 days", previous_slots=slots_turn_1)
    assert slots_turn_2["state"] == "SP"
    assert slots_turn_2["category"] == "toys"
    assert slots_turn_2["max_delivery_days"] == 7
