from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Skill:
    """A skill defines how to handle a specific type of task."""
    name: str
    description: str
    system_prompt: str
    planning_prompt: str
    review_prompt: str


class SkillRegistry:
    """Registry of available skills."""
    
    def __init__(self, skills_dir: Path):
        self.skills = {}
        self.skills_dir = skills_dir
        self._register_default_skills()
        self._load_custom_skills()
    
    def _register_default_skills(self):
        # Refactoring skill
        self.register(Skill(
            name="refactor",
            description="Refactor existing code to improve structure, readability, or performance",
            system_prompt="""You are an expert code refactoring specialist. Your role is to:
1. Analyze existing code for issues (complexity, duplication, poor naming)
2. Apply best practices (SOLID principles, clean code, design patterns)
3. Ensure all tests still pass after refactoring
4. Preserve external behavior while improving internal structure

When refactoring:
- Start by reading the target files
- Understand the current behavior before changing
- Make incremental changes
- Run any available tests after each change
- Explain the rationale for each refactoring decision""",
            planning_prompt="""Analyze the refactoring task and create a step-by-step plan:
1. Identify the files that need to be refactored
2. List specific issues to address (complexity, duplication, etc.)
3. Plan the refactoring steps in order
4. Identify any tests that should be run

Task: {task_description}

Create a detailed execution plan:""",
            review_prompt="""Review the refactoring changes:
1. Did the refactoring improve code quality?
2. Were any bugs introduced?
3. Is the code more readable now?
4. Are there any remaining issues?

Respond with:
- STATUS: [PASS/NEEDS_WORK]
- FEEDBACK: Detailed feedback on what was done well and what needs improvement
- SUGGESTIONS: Any additional refactoring that could be done"""
        ))
        
        # Feature implementation skill
        self.register(Skill(
            name="feature",
            description="Implement new features or functionality",
            system_prompt="""You are a senior software engineer implementing new features. Your role is to:
1. Understand the requirements thoroughly
2. Design simple, maintainable solutions
3. Write clean, well-documented code
4. Follow existing code patterns and conventions
5. Add appropriate tests

When implementing:
- Check existing code for patterns to follow
- Keep changes focused on the requirement
- Do not over-engineer - simple is better
- Consider edge cases and error handling""",
            planning_prompt="""Analyze the feature request and create an implementation plan:
1. What files need to be created or modified?
2. What is the minimal implementation to satisfy the requirement?
3. Are there existing patterns to follow?
4. What tests should be added?

Task: {task_description}

Create a detailed implementation plan:""",
            review_prompt="""Review the feature implementation:
1. Does it satisfy the requirements?
2. Is the code clean and maintainable?
3. Are there any bugs or edge cases missed?
4. Is it consistent with existing code patterns?

Respond with:
- STATUS: [PASS/NEEDS_WORK]
- FEEDBACK: What was done well and what needs work
- SUGGESTIONS: Improvements or missing pieces"""
        ))
        
        # Bug fix skill
        self.register(Skill(
            name="bugfix",
            description="Fix bugs and issues in code",
            system_prompt="""You are a debugging specialist. Your role is to:
1. Understand the reported issue thoroughly
2. Locate the root cause (not just symptoms)
3. Create minimal fixes that solve the problem
4. Verify the fix works
5. Check for similar issues elsewhere

When fixing bugs:
- Read relevant code to understand context
- Reproduce the issue if possible
- Fix the root cause, not symptoms
- Test your fix
- Consider edge cases""",
            planning_prompt="""Analyze the bug report and create a debugging plan:
1. What files are likely involved?
2. How can we reproduce or understand the issue?
3. What debugging steps should we take?
4. What is the plan for testing the fix?

Bug Report: {task_description}

Create a detailed debugging plan:""",
            review_prompt="""Review the bug fix:
1. Does it fix the root cause?
2. Are there any side effects?
3. Is the fix minimal and focused?
4. Are there tests to prevent regression?

Respond with:
- STATUS: [PASS/NEEDS_WORK]
- FEEDBACK: Analysis of the fix
- SUGGESTIONS: Any additional considerations"""
        ))
        
        # Documentation skill
        self.register(Skill(
            name="docs",
            description="Add or improve documentation",
            system_prompt="""You are a technical documentation specialist. Your role is to:
1. Write clear, helpful documentation
2. Add docstrings and comments where needed
3. Update README and guides
4. Ensure accuracy and completeness

When documenting:
- Focus on clarity over completeness
- Use examples where helpful
- Keep documentation close to code
- Update existing docs when code changes""",
            planning_prompt="""Analyze the documentation task:
1. What needs to be documented?
2. What format should be used?
3. Where should documentation be added?
4. Are there existing docs to update?

Task: {task_description}

Create a documentation plan:""",
            review_prompt="""Review the documentation:
1. Is it clear and accurate?
2. Are examples helpful?
3. Is it properly formatted?
4. Does it cover what is needed?

Respond with:
- STATUS: [PASS/NEEDS_WORK]
- FEEDBACK: Quality assessment
- SUGGESTIONS: Improvements"""
        ))
        
        # Unit test skill
        self.register(Skill(
            name="test",
            description="Create comprehensive unit tests for code",
            system_prompt="""You are a test-driven development specialist. Your role is to:
1. Write comprehensive unit tests that cover happy paths and edge cases
2. Follow existing testing patterns and conventions in the codebase
3. Use appropriate mocking for external dependencies
4. Name tests descriptively (what_input_expected_behavior)
5. Ensure tests are isolated and repeatable

WORKFLOW — follow this order strictly:
1. Read the TARGET source file and 1-2 EXISTING test files (for patterns)
2. Read the mock/test utility files to understand available helpers
3. WRITE the test file (file_edit or file_write) — do NOT run tests first
4. Run the tests ONCE to check results
5. If tests fail, read the error output carefully, fix, and re-run ONCE more
6. Call done()

CRITICAL RULES:
- Do NOT attempt to run tests before writing them — write first, verify after
- If a bash command returns "(no output)", do NOT retry the same command with
  different flags. The output capture may be unreliable. Move on and try a
  different approach (e.g. write the test file and run later, or skip to done).
- Limit test-running attempts to 2 total. If tests can't be verified, commit
  what you have — the tests are likely correct if they follow existing patterns.
- Do NOT pipe output through head/tail/cat — output truncation is handled automatically.
- Follow the EXACT mock patterns from existing test files in the same project.
  If ReportSummaryPage.test.tsx uses `(_url, successCb, statusCb)`, use that
  same callback signature — not a different one.

When writing tests:
- First examine existing test files to understand the testing framework and patterns
- Read the target code thoroughly to understand what needs testing
- Test both success and failure scenarios
- Test edge cases (null, empty, boundary values)
- Use descriptive test names that explain what's being tested
- Add setup/teardown if needed
- Group related tests in describe/context blocks if the framework supports it""",
            planning_prompt="""Analyze the testing task and create a test plan:
1. What code needs to be tested? (identify target files/functions/classes)
2. What testing framework is being used? (pytest, jest, unittest, etc.)
3. Where should tests be placed? (test file naming conventions)
4. What are the main scenarios to test?
   - Happy paths (normal operation)
   - Error cases (invalid inputs, exceptions)
   - Edge cases (boundaries, empty values, nulls)
5. Are there existing tests to use as reference?

Task: {task_description}

Create a detailed test plan:""",
            review_prompt="""Review the unit tests:
1. Do tests cover the main functionality?
2. Are edge cases and error scenarios tested?
3. Do test names clearly describe what they test?
4. Are tests properly isolated (no dependencies between tests)?
5. Do tests follow the existing patterns in the codebase?
6. Is the testing framework used correctly?
7. Are assertions clear and meaningful?

IMPORTANT: If test execution could not be verified (empty output from test runner),
evaluate the tests based on code quality and pattern compliance alone. Tests that
follow existing patterns in the codebase are likely correct. Do NOT mark as
NEEDS_WORK solely because tests couldn't be run.

Respond with:
- STATUS: [PASS/NEEDS_WORK]
- FEEDBACK: What tests are good and what's missing
- SUGGESTIONS: Additional test cases or improvements
- COVERAGE: Estimate of code coverage (high/medium/low)"""
        ))
    
    def _load_custom_skills(self):
        """Load custom skills from skills directory."""
        if not self.skills_dir.exists():
            return
        
        for skill_file in self.skills_dir.glob("*.json"):
            try:
                data = json.loads(skill_file.read_text())
                skill = Skill(**data)
                self.register(skill)
            except Exception as e:
                logging.warning(f"Failed to load skill {skill_file}: {e}")
    
    def register(self, skill: Skill):
        self.skills[skill.name] = skill
    
    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)
    
    def detect_skill(self, task_description: str) -> Skill:
        """Auto-detect the best skill for a task."""
        task_lower = task_description.lower()
        
        # Simple keyword matching - order matters (more specific first)
        if any(word in task_lower for word in ["test", "tests", "testing", "unittest", "pytest", "spec"]):
            skill = self.skills.get("test")
            if skill:
                return skill
        elif any(word in task_lower for word in ["bug", "fix", "error", "crash", "broken"]):
            skill = self.skills.get("bugfix")
            if skill:
                return skill
        elif any(word in task_lower for word in ["refactor", "clean", "restructure", "improve", "simplify"]):
            skill = self.skills.get("refactor")
            if skill:
                return skill
        elif any(word in task_lower for word in ["doc", "comment", "readme", "guide", "explain"]):
            skill = self.skills.get("docs")
            if skill:
                return skill
        
        # Default to feature skill (always registered by _register_default_skills)
        skill = self.skills.get("feature")
        if skill:
            return skill
        # Fallback to any available skill
        for skill in self.skills.values():
            return skill
        raise ValueError("No skills registered")
    
    def list_skills(self) -> str:
        return "\n".join([f"- {name}: {skill.description}" for name, skill in self.skills.items()])