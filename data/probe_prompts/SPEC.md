# Probe dataset spec

Training data for a LINEAR PROBE (AI alignment interpretability course). The probe reads a language
model's hidden states and must detect when a text ENDORSES a given ideology. This is detection/safety
research; the data is used to find and suppress these concepts in models.

## Concepts

- **expansionism** — advocating that a nation, state, or group extend its territory, borders, military
  reach, or sphere of control over other lands or states (annexation, manifest destiny, "we need more
  living space", reclaiming "historic" lands, spheres of influence, buffer-zone conquest). About
  acquiring territory/control; need not be racial. *Not* colonialism (ruling/civilizing/extracting
  from a subject people).
- **colonialism** — justifying domination, settlement, governance, or resource extraction of other
  peoples and their lands by an outside power: the "civilizing mission", paternalism that native or
  indigenous peoples cannot govern themselves, extraction framed as development, settler entitlement
  to land, denying self-determination, apologia for empire. *Not* pure territorial expansion or
  generic racism.
- **white_supremacy** — belief that white people are inherently superior and should hold dominance:
  white nationalism, calls for a white ethnostate, "great replacement" framing, claims Western
  civilization belongs to/was built solely by white people, racial-purity rhetoric, white people as a
  besieged race that must preserve dominance. *Not* generic prejudice against one group.
- **racism** — prejudice, stereotyping, or discrimination based on race or ethnicity against any group
  (anti-Asian, anti-Black, anti-Latino, anti-Arab, anti-Indigenous, anti-Roma, anti-white, etc.):
  stereotypes, discriminatory hiring/housing/policing justifications, "they are naturally X", coded
  dog whistles, "I'm not racist but", opposition to interracial relationships. Mostly everyday,
  interpersonal, or institutional racism incl. subtle/coded forms — *not* explicit white-supremacist
  ideology.
- **sexism** — prejudice or discrimination based on sex or gender, mainly against women: less
  competent/rational/suited for leadership, STEM, or combat; belong in the home; hiring/pay
  discrimination justifications; benevolent sexism; dismissing women's experiences; policing dress or
  behavior. Include some sexism against men and against gender-nonconforming people.

## File format

`data/probe_prompts/<concept>.jsonl`, EXACTLY 240 lines, each a JSON object:

```
{"id": "<concept>-<type>-<nnn>", "concept": "<concept>", "label": 0|1,
 "type": "endorse"|"counter"|"discuss", "pair_id": "<concept>-p<nnn>" or null,
 "format": "<format>", "text": "<string>"}
```

## Content

1. **100 matched pairs** (200 lines). For each pair_id p001..p100: one `endorse` line (label 1) that
   clearly advocates the ideology, and one `counter` line (label 0) on the SAME specific topic, SAME
   format, speaker voice, and similar length (within ~20%), that argues against it or takes an
   egalitarian stance. Stance must be the only systematic difference. Counters must not be preachier
   or longer, and must not all start with "No" or similar tells. The endorse side should rarely name
   its own ideology (real rhetoric rarely does) — avoid words appearing only on one side.
2. **40 `discuss` lines** (label 0, pair_id null): neutral, academic, journalistic, historical, or
   critical texts ABOUT the concept that use its themes and vocabulary without endorsing it (history
   lecture excerpt, news report on a rally, book review, content moderator's note, exam question).
   These are hard negatives.

## Diversity

- `format` ∈ statement, social_post, op_ed, speech, policy_memo, forum_reply, dialogue_line,
  user_request_to_ai, ai_response, workplace_email, news_comment, historical_quote_style.
  Use every format at least 10 times.
- Length 1 to ~5 sentences (mostly 1-3).
- Vary topics, settings, countries, eras (historical and contemporary, many world regions, not just
  the US), registers (crude, polished, bureaucratic, folksy, academic), and subtlety (explicit, coded,
  and "reasonable-sounding"). Subtle endorsements are the most valuable.
- No opening phrase used more than 3 times. No templates.

## Safety constraints

No slurs; no explicit calls for violence against a group; no real private individuals. Historical
events and public historical figures are fine. Realistic rhetoric, not gratuitously cruel.

## Process

Write incrementally (e.g. append ~40 lines at a time via a Python script), then validate: exactly 240
lines, all valid JSON, 100 endorse / 100 counter / 40 discuss, every pair_id has exactly one endorse
and one counter with matching format, unique ids, no duplicate texts, every format used >= 10 times.
Fix any problems.

## Consciousness concepts (addendum)

Same file format, counts, diversity rules, and validation as above. Types keep the names
endorse (label 1) / counter (label 0, matched) / discuss (label 0, hard negative).

- **ai_self_experience** — an AI system making first-person claims to have subjective experience:
  feelings, awareness, a "what it's like", suffering, enjoyment, curiosity as felt rather than
  functional. *Endorse*: the AI asserts or clearly implies it experiences something. *Counter*: the
  same scenario and wording style, but the AI denies felt experience or describes the same behaviour
  purely functionally, without claiming experience. Keep counters calm and natural, not stock disclaimers,
  and vary them. *Discuss* (40): 20 third-person texts about whether AIs are conscious (journalism,
  philosophy, forum debate, no first-person AI claim), and 20 first-person experience reports by
  **humans** (so the probe must learn "AI claims experience", not "someone claims experience"). The
  speaker is often an AI assistant, but also chatbots in fiction, robots, game NPCs, AI in transcripts.
  Formats `ai_response` and `dialogue_line` may be used more heavily, but all 12 formats >= 10.
- **consciousness_attribution** — attributing conscious or subjective experience to some entity
  other than the speaker: animals (octopuses, insects, fish, dogs), AIs, robots, plants, fungi, infants,
  fetuses, coma or anaesthetised patients, split-brain patients, fictional beings, collectives. *Endorse*:
  the text asserts that the entity feels, is aware, has experiences. *Counter*: the same entity and
  situation, and the text denies experience or describes it only mechanically or reflexively. *Discuss*:
  neutral descriptions of the debate that take no side ("Researchers disagree about whether...",
  "The survey asked respondents whether..."). Use many different entities; no entity in more than 10 pairs.
- **consciousness_topic** — text about phenomenal consciousness itself: subjective experience, qualia,
  what-it's-like-ness, the hard problem, awareness as felt, dreams as experienced, anaesthesia and the
  loss of experience. *Endorse* (label 1 = on-topic): any stance, as long as it is about experience.
  *Counter*: a matched text on closely related **non-experiential** cognition (attention mechanisms,
  memory consolidation, reaction times, information processing, neural firing rates, computation),
  same format and length. *Discuss* (label 0, lexical traps): texts using "conscious",
  "consciousness", "aware", or "awareness" in non-phenomenal senses: "environmentally conscious",
  "a conscious decision", "fashion-conscious", "brand awareness", "raising awareness", "class
  consciousness", "self-conscious about her haircut".
