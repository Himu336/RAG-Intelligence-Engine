# Interview System Optimization Plan

## Current State
- **API Calls**: 7 calls per interview (5 questions)
  - 1 call: Question generation
  - 5 calls: Individual answer evaluations (sequential)
  - 1 call: Overall analysis
- **Token Usage**: ~9,500 tokens average
- **Flow**: Sequential processing

## Optimization Goals
- Reduce API calls by 70%+ (7 → 2 calls)
- Reduce token usage by 30-40%
- Maintain same user experience and flow
- Keep same analysis quality

## Proposed Optimizations

### 1. **Batch Answer Evaluation** (Major Optimization)
**Current**: Evaluate each answer in separate API calls (5 calls)
**Optimized**: Evaluate all answers in single batch call (1 call)

**Benefits**:
- Reduces 5 calls → 1 call (80% reduction in evaluation calls)
- Eliminates redundant prompt overhead (~400 tokens × 4 = 1,600 tokens saved)
- Faster processing (parallel evaluation vs sequential)
- Better context for overall analysis (LLM sees all answers at once)

**Implementation**:
- Create `_evaluate_all_answers_batch()` method
- Single prompt with all Q&A pairs
- Parse array of evaluations from single JSON response

**Token Savings**: ~1,600-2,000 tokens

---

### 2. **Combine Evaluation + Overall Analysis** (Major Optimization)
**Current**: Separate calls for evaluations and overall analysis
**Optimized**: Generate both in same batch call

**Benefits**:
- Reduces 1 call → 0 (merged with batch evaluation)
- LLM has full context for better overall analysis
- Eliminates redundant prompt (~800 tokens saved)

**Implementation**:
- Include overall analysis request in batch evaluation prompt
- Return both individual evaluations and overall analysis in single response

**Token Savings**: ~800-1,200 tokens

---

### 3. **Optimize Prompt Efficiency** (Minor Optimization)
**Current**: Verbose prompts with redundant instructions
**Optimized**: Concise, focused prompts

**Changes**:
- Remove redundant instructions
- Use shorter evaluation criteria
- Limit context to essentials
- Shorter system messages

**Token Savings**: ~500-800 tokens per batch call

---

### 4. **Smart Answer Truncation** (Minor Optimization)
**Current**: Send full answers (can be very long)
**Optimized**: Truncate very long answers intelligently

**Implementation**:
- Keep first 500 words + last 100 words (for context)
- Or summarize if > 1000 words
- Preserve key information while reducing tokens

**Token Savings**: ~200-500 tokens (depends on answer length)

---

## Optimized State

### API Calls
- **Before**: 7 calls
- **After**: 2 calls
  - 1 call: Question generation
  - 1 call: Batch evaluation + overall analysis
- **Reduction**: 71% fewer calls

### Token Usage
- **Before**: ~9,500 tokens
- **After**: ~6,000-6,500 tokens
- **Reduction**: 30-35% fewer tokens

### Breakdown (Optimized):
1. **Question Generation**: ~2,500 tokens (unchanged)
2. **Batch Evaluation + Analysis**: ~3,500-4,000 tokens
   - Input: ~2,000 tokens (all Q&A pairs + instructions)
   - Output: ~1,500-2,000 tokens (all evaluations + overall analysis)

---

## Implementation Priority

### Phase 1: Batch Evaluation (High Impact)
- Implement `_evaluate_all_answers_batch()`
- Replace loop in `generate_analysis()`
- **Impact**: 5 calls → 1 call, ~1,600 tokens saved

### Phase 2: Combine with Overall Analysis (High Impact)
- Merge overall analysis into batch call
- **Impact**: 1 call → 0, ~800 tokens saved

### Phase 3: Prompt Optimization (Medium Impact)
- Refine prompts for conciseness
- **Impact**: ~500 tokens saved

### Phase 4: Answer Truncation (Low Impact)
- Add smart truncation
- **Impact**: ~200-500 tokens saved (variable)

---

## Expected Results

### Cost Reduction
- **API Calls**: 71% reduction (7 → 2)
- **Tokens**: 30-35% reduction (~9,500 → ~6,500)
- **Cost**: ~40-45% reduction overall

### Performance
- **Speed**: Faster (1 batch call vs 6 sequential calls)
- **Quality**: Same or better (LLM sees full context)
- **Reliability**: Better (fewer API calls = fewer failure points)

### User Experience
- **No changes**: Same flow, same endpoints, same responses
- **Faster analysis**: Batch processing is quicker
- **Same quality**: Analysis quality maintained or improved

---

## Risks & Considerations

1. **Token Limits**: Batch call needs higher `max_output_tokens` (3000-4000)
2. **JSON Parsing**: More complex JSON structure to parse
3. **Error Handling**: If batch fails, need fallback strategy
4. **Testing**: Ensure quality matches current implementation

---

## Next Steps

1. Implement batch evaluation method
2. Update `generate_analysis()` to use batch
3. Test with various interview scenarios
4. Monitor token usage and quality
5. Optimize prompts iteratively

