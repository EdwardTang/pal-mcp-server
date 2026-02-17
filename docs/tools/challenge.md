# challenge - Challenge an approach or validate ideas with confidence

The `challenge` tool encourages thoughtful critical thinking instead of automatic agreement with the dreaded **You're absolutely right!** responses - especially 
when you're not. This tool wraps your comment with instructions that prompt critical thinking and honest analysis instead of blind agreement.

It now uses a multi-phase protocol inspired by multi-agent orchestration:
1. Decompose claim and assumptions.
2. Generate competing hypotheses.
3. Stress-test with domain lenses (risk, evidence quality, performance/security/cost when relevant).
4. Return verdict with confidence and decision-changing evidence.

## Quick Example

```
challenge but do we even need all this extra caching because it'll just slow the app down?
```

```
challenge I don't think this approach solves my original complaint
```

Normally, your favorite coding agent will enthusiastically reply with **“You’re absolutely right!”**—then proceed to 
reverse the _correct_ strategy entirely, without stopping to consider that you might actually be wrong, missing the 
bigger picture or ignoring architectural constraints.

`challenge` fixes this. Claude can even _detect_ when you're challenging something and automatically invokes this tool
to ensure thoughtful analysis instead of reflexive agreement.

**Without PAL:**
![without_pal@2x](https://github.com/user-attachments/assets/64f3c9fb-7ca9-4876-b687-25e847edfd87)

**With PAL:**
![with_pal@2x](https://github.com/user-attachments/assets/9d72f444-ba53-4ab1-83e5-250062c6ee70)

## Why Use Challenge?

AI assistants sometimes tend to agree too readily. The challenge tool helps you:
- Get genuine critical evaluation of your ideas
- Challenge assumptions constructively
- Receive honest feedback on proposals
- Validate approaches with thoughtful analysis

## Benchmarking

Challenge quality is included in the consensus/challenge A/B harness:

```bash
python scripts/benchmark_consensus_challenge_ab.py --iterations 100
```
