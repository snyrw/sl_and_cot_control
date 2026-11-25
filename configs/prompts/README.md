# Prompts

- `elicited_cotcontrol_v1.txt` — elicitation suffix appended (after a blank line) to the
  CoT-Control user prompt in the `elicited` regime. Written by us from the public
  *description* of Thomas Read's UK AISI prompt (LessWrong shortform): apply the constraint
  from the very first private thought, to thinking only, drop it when answering, never
  mention it. The original text is not public, so this is **not** a reproduction of it.
  `{reasoning_term}` is filled with "analysis channel" for gpt-oss, "reasoning" otherwise.
  Never edit a versioned file after results exist; add `_v2` instead (the text enters cache
  keys via the rendered prompt, so edits silently fork caches).
