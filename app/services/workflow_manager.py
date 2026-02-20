"""Workflow manager to handle workflow questions in sequence with branching support"""
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger("assistly.workflow_manager")


class WorkflowManager:
    """Manages workflow question flow with optional branching via multiple-choice options"""
    
    def __init__(self, context: Dict[str, Any]):
        self.context = context
        self.current_treatment_plan: Optional[str] = None
        self.current_workflow: Optional[Dict[str, Any]] = None
        self.current_question_index: int = 0
        # When branching, we track an explicit ordered list of question IDs to ask
        self._question_queue: Optional[List[Dict[str, Any]]] = None
        self._queue_index: int = 0
        self.workflow_answers: Dict[str, Any] = {}
        self.is_active: bool = False
        # Base URL for attachment download links (set from config/context)
        self.api_base_url: str = context.get("api_base_url", "")
    
    def start_workflow_for_treatment_plan(self, treatment_plan_name: str) -> bool:
        """Start workflow for a treatment plan. Returns True if workflow exists, False otherwise."""
        self.current_treatment_plan = treatment_plan_name
        
        treatment_plans = self.context.get("treatment_plans", [])
        logger.info(f"Looking for treatment plan '{treatment_plan_name}' in {len(treatment_plans)} treatment plans")
        treatment_plan = None
        for tp in treatment_plans:
            if isinstance(tp, dict):
                tp_question = tp.get("question", "")
                if tp_question.lower() == treatment_plan_name.lower():
                    treatment_plan = tp
                    logger.info(f"✓ Found treatment plan: '{tp_question}' with keys: {list(tp.keys())}")
                    break
        
        if not treatment_plan:
            logger.warning(f"✗ Treatment plan '{treatment_plan_name}' not found in context.")
            return False
        
        attached_workflows = treatment_plan.get("attachedWorkflows", [])
        logger.info(f"Treatment plan has {len(attached_workflows)} attached workflows")
        if not attached_workflows:
            return False
        
        workflow_id = attached_workflows[0].get("workflowId")
        if not workflow_id:
            return False
        
        workflows = self.context.get("workflows", [])
        workflow = None
        for wf in workflows:
            if wf.get("_id") == workflow_id:
                workflow = wf
                break
        
        if not workflow:
            logger.warning(f"✗ Workflow '{workflow_id}' not found in context.")
            return False
        
        questions = workflow.get("questions", [])
        if not questions:
            return False
        
        sorted_questions = sorted(
            [q for q in questions if q.get("isActive", True)],
            key=lambda q: q.get("order", 0)
        )
        
        if not sorted_questions:
            return False
        
        self.current_workflow = workflow
        self.current_question_index = 0
        self._question_queue = sorted_questions
        self._queue_index = 0
        self.workflow_answers = {}
        self.is_active = True
        
        logger.info(
            f"Started workflow '{workflow_id}' for '{treatment_plan_name}' "
            f"with {len(sorted_questions)} questions"
        )
        return True
    
    def _get_sorted_questions(self) -> List[Dict[str, Any]]:
        """Return the ordered question list (either queue or sorted from workflow)"""
        if self._question_queue is not None:
            return self._question_queue
        if not self.current_workflow:
            return []
        questions = self.current_workflow.get("questions", [])
        return sorted(
            [q for q in questions if q.get("isActive", True)],
            key=lambda q: q.get("order", 0)
        )
    
    def get_current_question(self) -> Optional[Dict[str, Any]]:
        """Get the current question to ask"""
        if not self.is_active or not self.current_workflow:
            return None
        
        questions = self._get_sorted_questions()
        idx = self._queue_index if self._question_queue is not None else self.current_question_index
        
        if idx >= len(questions):
            return None
        
        return questions[idx]
    
    def get_question_attachment_url(self, question: Dict[str, Any]) -> Optional[str]:
        """Return a public download URL if the question has an attached file."""
        attachment = question.get("attachment", {})
        if not attachment.get("hasFile"):
            return None
        
        question_id = question.get("_id")
        
        # Resolve app_id from multiple possible locations in context
        app_id = (
            (self.current_workflow or {}).get("owner")
            or (self.context.get("app") or {}).get("_id")
            or (self.context.get("app") or {}).get("id")
            or (self.context.get("integration") or {}).get("appId")
        )
        
        if not question_id or not app_id:
            return None
        
        base = self.api_base_url.rstrip("/")
        if not base:
            return None
        
        return f"{base}/chatbot-workflows/apps/{app_id}/{question_id}/attachment"
    
    def format_question_with_options(self, question: Dict[str, Any]) -> str:
        """
        Build the bot message for a question, including:
        - Option buttons for multiple-choice (branching)
        - A download button if there's an attached file
        """
        q_text = question.get("question", "")
        options = question.get("options", []) or []
        
        parts = [q_text]
        
        # Attach file download link if available
        attachment_url = self.get_question_attachment_url(question)
        filename = (question.get("attachment") or {}).get("filename", "Download file")
        if attachment_url:
            parts.append(
                f'<file url="{attachment_url}" name="{filename}">📎 {filename}</file>'
            )
        
        # Add multiple-choice option buttons
        if options:
            sorted_opts = sorted(options, key=lambda o: o.get("order", 0))
            for opt in sorted_opts:
                opt_text = opt.get("text", "").strip()
                if opt_text:
                    parts.append(f"<button>{opt_text}</button>")
        
        return "\n".join(parts)
    
    def record_answer(self, answer: str) -> bool:
        """
        Record answer to current question and advance.
        For multiple-choice questions with branching, jump to the linked next question.
        Returns True if more questions remain.
        """
        if not self.is_active:
            return False
        
        current_question = self.get_current_question()
        if not current_question:
            return False
        
        question_id = current_question.get("_id")
        question_text = current_question.get("question", "")
        options = current_question.get("options", []) or []
        
        # Store answer
        self.workflow_answers[question_id] = {
            "question": question_text,
            "answer": answer,
            "order": current_question.get("order", 0)
        }
        
        # Determine next question for branching
        next_question_id = self._resolve_next_question(answer, options)
        
        if next_question_id:
            # Branching: find target question and add it next in the queue
            logger.info(f"Branching to question '{next_question_id}' after answer '{answer}'")
            target_q = self._find_question_by_id(next_question_id)
            if target_q:
                if self._question_queue is not None:
                    # Insert target after current position
                    insert_at = self._queue_index + 1
                    # Remove if already in queue to avoid duplicates
                    self._question_queue = [q for q in self._question_queue if q.get("_id") != next_question_id]
                    self._question_queue.insert(insert_at, target_q)
                    self._queue_index += 1
                else:
                    # Convert to queue-based tracking
                    sorted_qs = self._get_sorted_questions()
                    self._question_queue = sorted_qs[:]
                    insert_at = self._queue_index + 1
                    self._question_queue = [q for q in self._question_queue if q.get("_id") != next_question_id]
                    self._question_queue.insert(insert_at, target_q)
                    self._queue_index += 1
            else:
                self._advance_index()
        else:
            self._advance_index()
        
        logger.info(
            f"Recorded answer for '{question_text}': '{answer}'. "
            f"Queue index now {self._queue_index} of {len(self._get_sorted_questions())}"
        )
        
        return self.get_current_question() is not None
    
    def _advance_index(self):
        """Move to next question sequentially"""
        if self._question_queue is not None:
            self._queue_index += 1
        else:
            self.current_question_index += 1
    
    def _resolve_next_question(self, answer: str, options: List[Dict[str, Any]]) -> Optional[str]:
        """
        Given the user's answer and the question's options, find the nextQuestionId to branch to.
        Matches the selected option by text (case-insensitive).
        """
        if not options:
            return None
        
        answer_lower = answer.strip().lower()
        for opt in options:
            opt_text = (opt.get("text") or "").strip()
            if opt_text.lower() == answer_lower:
                next_id = opt.get("nextQuestionId")
                if opt.get("isTerminal"):
                    # Terminal option – end the workflow
                    logger.info(f"Option '{opt_text}' is terminal, ending workflow")
                    self._end_workflow()
                    return None
                return next_id if next_id else None
        
        return None
    
    def _end_workflow(self):
        """Force workflow to end (terminal option selected)"""
        if self._question_queue is not None:
            self._queue_index = len(self._question_queue)
        else:
            questions = self._get_sorted_questions()
            self.current_question_index = len(questions)
    
    def _find_question_by_id(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Search for a question in the current workflow by _id"""
        if not self.current_workflow:
            return None
        questions = self.current_workflow.get("questions", [])
        for q in questions:
            if q.get("_id") == question_id:
                return q
        return None
    
    def _get_total_questions(self) -> int:
        """Get total number of active questions in current workflow"""
        return len(self._get_sorted_questions())
    
    def is_workflow_complete(self) -> bool:
        """Check if all workflow questions have been answered"""
        if not self.is_active:
            return True
        return self.get_current_question() is None
    
    def get_workflow_answers(self) -> Dict[str, Any]:
        """Get all workflow answers"""
        return self.workflow_answers.copy()
    
    def reset(self):
        """Reset workflow state"""
        self.current_treatment_plan = None
        self.current_workflow = None
        self.current_question_index = 0
        self._question_queue = None
        self._queue_index = 0
        self.workflow_answers = {}
        self.is_active = False
