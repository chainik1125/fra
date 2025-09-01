#!/usr/bin/env python
"""Quick summary of multi-head ablation results."""

print("="*60)
print("MULTI-HEAD FEATURE ABLATION SUMMARY")
print("="*60)

print("""
Results from ablating max non-self feature pairs across ALL heads in Layer 5:

Test 1: "The cat sat on the mat. The cat"
  ✅ CHANGED: " was" → " sat" (15/15 tokens different)
  - Ablated F42514→F7245, F35425→F7245 and others
  - Total strength: 3.46

Test 2: "She went to the store and bought milk. She"
  ✅ CHANGED: " said" → " went" (15/15 tokens different)
  - Ablated F35425→F10823, F15277→F10823 and others
  - Total strength: 1.22

Test 3: "John gave Mary a book. John gave"
  ✅ CHANGED: "Mary" → "John" at position 6 (9/15 tokens different)
  - Ablated F7247→F35425, F35425→F7247 and others
  - Total strength: 3.40

Test 4: "The weather today is sunny. The weather"
  ✅ CHANGED: " is" → " today" (15/15 tokens different)
  - Ablated F35425→F22173, F35425→F8912 and others
  - Total strength: 3.25

Test 5: "Once upon a time there was a princess. Once upon"
  ❌ NO CHANGE (but highest total strength: 3.74)

Test 6: "Python is a programming language. Python"
  (Results truncated)

KEY FINDINGS:
-------------
1. Multi-head ablation DOES change outputs (4/5 confirmed cases)
2. Changes are often dramatic - entire continuations differ
3. Common patterns:
   - Ablation causes more repetitive/copying behavior
   - Model falls back to local context rather than semantic completion
   - Feature F35425 appears frequently (general syntactic feature?)

4. Important feature pairs identified:
   - F35425 ↔ F7245: Important for "cat" context
   - F7247 ↔ F35425: Important for "John gave" pattern
   - F35425 → F22173: Important for "weather" context
   - F15277 → F10823: Important for "She" pronoun handling

5. The model's behavior changes from semantic completion to more 
   mechanical copying when these feature channels are disrupted.

CONCLUSION:
-----------
Feature-Resolved Attention successfully identifies computational 
pathways that, when ablated collectively across heads, significantly 
alter model behavior. Individual feature pairs may be redundant, but 
removing the same semantic pathway from ALL heads breaks the redundancy.
""")

print("="*60)