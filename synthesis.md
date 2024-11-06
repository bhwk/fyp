ok so limiting ourselves to small models brings some restrictions:

- We have to create multiple LLM agents with separate prompts to do what we need it to do
- Streaming sentences is a must. Passing the entirety of the text in one batch causes the model to lose the system prompt context
- Dates should probably be handled by function calling rather than on the LLM's ability
- Small LLMs can round numbers properly, but asking them to change that exact number to a range is difficult
- Providing an output format increases consistency in output
- need to pick an optimal number of observations to pass to the LLM without it losing context
- for some reason the v0.2 mistral instruct is different from the latest instruct???? both are running q8 so this doesn't make sense
