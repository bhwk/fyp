# Todo

- [x] TEST OUT FUNCTIONS WITH AGENT # IMPORTANT
  - [ ] Ok now figure out how to reliably guide the LLM to make use of the functions properly
- [x] Find a way to relate synth context to original query without confusing the model -- Done by generating a synth query, and returning it + synth context to LLM
- [ ] Make the LLM able to plan out steps for queries that involve multiple patients
- [x] ~~Use metadata extraction to possibly improve vector search~~ not a great idea on second thought, can't reliably extract entities
- [ ] Figure out a workflow for agents, create appropriate functions to carry out retrieval, synth, and response # IN PROGRESS
  - [ ] !!! Combine the query-planning workflow + reflection workflow from llamaindex
- [ ] Look into using the multi-agent-concierge framework
