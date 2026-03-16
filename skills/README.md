# Skills Folder

This folder is for **custom** skills only (`*.json` files).

The following skills are already built into the agent and do **not** need to be added here:

- `refactor`
- `feature`
- `bugfix`
- `docs`
- `test`

If you add a custom skill with one of those same names, it will override the built-in version.

## Custom Skill Format

Create JSON files like:

```json
{
  "name": "performance",
  "description": "Optimize code for better performance",
  "system_prompt": "You are a performance optimization specialist...",
  "planning_prompt": "Analyze performance bottlenecks and create optimization plan...",
  "review_prompt": "Review the performance improvements..."
}
```
