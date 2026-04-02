"""Workflow manager to handle workflow questions in sequence with branching support"""
from typing import Dict, Any, Optional, List, Tuple
import logging
import re

logger = logging.getLogger("assistly.workflow_manager")


class WorkflowManager:
    """Manages workflow question flow with optional branching via multiple-choice options"""
    
    def __init__(self, context: Dict[str, Any]):
        self.context = context
        # Note: conversationStyle must be read dynamically from context so that
        # toggle changes can apply to existing sessions without recreating this manager.
        integration = (context.get("integration") or {}) if isinstance(context, dict) else {}
        self.conversation_style: bool = bool(integration.get("conversationStyle"))
        self.current_service_name: Optional[str] = None
        self.current_workflow: Optional[Dict[str, Any]] = None
        self.current_question_index: int = 0
        # When branching, we track an explicit ordered list of question IDs to ask
        self._question_queue: Optional[List[Dict[str, Any]]] = None
        self._queue_index: int = 0
        self.workflow_answers: Dict[str, Any] = {}
        self.is_active: bool = False
        # IDs of questions that are branch targets — excluded from sequential queue
        self._linked_question_ids: set = set()
        # Base URL for attachment download links (set from config/context)
        self.api_base_url: str = context.get("api_base_url", "")

    def _conversation_style_enabled(self) -> bool:
        """Read conversational style flag from the latest context."""
        integration = self.context.get("integration") or {}
        if not isinstance(integration, dict):
            return False
        value = integration.get("conversationStyle")
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _normalize_id(self, value: Any) -> str:
        """Normalize IDs to comparable strings across payload shapes."""
        if value is None:
            return ""
        return str(value).strip()
    
    def start_workflow_for_service(self, service_name: str) -> bool:
        """Start workflow for a service/menu item. Returns True if workflow exists, False otherwise."""
        self.current_service_name = service_name

        service_plans = self.context.get("service_plans", [])
        logger.info(f"Looking for service '{service_name}' in {len(service_plans)} service plans")
        service_plan = None
        for tp in service_plans:
            if isinstance(tp, dict):
                tp_question = tp.get("question", "")
                if tp_question.lower() == service_name.lower():
                    service_plan = tp
                    logger.info(f"✓ Found service plan: '{tp_question}' with keys: {list(tp.keys())}")
                    break

        if not service_plan:
            logger.warning(f"✗ Service '{service_name}' not found in context.")
            return False

        attached_workflows = service_plan.get("attachedWorkflows", [])
        logger.info(f"Service plan has {len(attached_workflows)} attached workflows")
        if not attached_workflows:
            return False
        
        workflow_id = self._normalize_id(attached_workflows[0].get("workflowId"))
        if not workflow_id:
            return False
        
        workflows = self.context.get("workflows", [])
        workflow = None
        for wf in workflows:
            if self._normalize_id(wf.get("_id")) == workflow_id:
                workflow = wf
                break
        
        if not workflow:
            logger.warning(f"✗ Workflow '{workflow_id}' not found in context.")
            return False
        
        questions = workflow.get("questions", [])
        if not questions:
            return False

        # Collect branch-only targets:
        # - If nextQuestionId simply points to the immediate next ordered question,
        #   treat it as linear progression (NOT branch-only).
        # - If it jumps elsewhere, treat target as branch-only and exclude it from
        #   the default sequential queue until that branch is selected.
        ordered_questions = sorted(questions, key=lambda q: q.get("order", 0))
        sequence_ids = [self._normalize_id(q.get("_id")) for q in ordered_questions if q.get("_id")]
        sequence_index_map = {qid: idx for idx, qid in enumerate(sequence_ids)}
        linked_question_ids: set = set()
        for q in questions:
            source_id = self._normalize_id(q.get("_id"))
            source_index = sequence_index_map.get(source_id)
            for opt in (q.get("options") or []):
                nqid = self._normalize_id(opt.get("nextQuestionId"))
                if not nqid:
                    continue
                target_index = sequence_index_map.get(nqid)
                # If branch target is not the immediate next sequential question,
                # treat it as branch-only.
                if source_index is None or target_index is None or target_index != source_index + 1:
                    linked_question_ids.add(nqid)

        logger.info(f"Workflow branch-only linked question IDs: {linked_question_ids}")

        active_questions = [q for q in questions if q.get("isActive", True)]
        # Only sequential questions go into the initial queue
        sequential_questions = sorted(
            [q for q in active_questions if self._normalize_id(q.get("_id")) not in linked_question_ids],
            key=lambda q: q.get("order", 0)
        )

        if not sequential_questions:
            return False

        self.current_workflow = workflow
        self._linked_question_ids = linked_question_ids   # kept for reference
        self.current_question_index = 0
        self._question_queue = sequential_questions
        self._queue_index = 0
        self.workflow_answers = {}
        self.is_active = True
        
        logger.info(
            "Workflow queue initialized: %s",
            [
                {
                    "id": self._normalize_id(q.get("_id")),
                    "order": q.get("order", 0),
                    "question": (q.get("question", "")[:80] + "...") if len(q.get("question", "")) > 80 else q.get("question", ""),
                }
                for q in sequential_questions
            ],
        )
        logger.info(
            f"Started workflow '{workflow_id}' for service '{service_name}' "
            f"with {len(sequential_questions)} sequential + {len(linked_question_ids)} linked questions"
        )
        return True

    def _get_sorted_questions(self) -> List[Dict[str, Any]]:
        """Return the ordered question list (either the live queue or the sequential subset)."""
        if self._question_queue is not None:
            return self._question_queue
        if not self.current_workflow:
            return []
        questions = self.current_workflow.get("questions", [])
        return sorted(
            [
                q for q in questions
                if q.get("isActive", True) and self._normalize_id(q.get("_id")) not in self._linked_question_ids
            ],
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
        
        # Add multiple-choice option buttons (disabled in conversational mode)
        if options and not self._conversation_style_enabled():
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
        next_question_id, end_after_branch = self._resolve_next_question(answer, options)
        
        if next_question_id:
            # Branching: find target question and add it next in the queue
            logger.info(f"Branching to question '{next_question_id}' after answer '{answer}' (terminal={end_after_branch})")
            target_q = self._find_question_by_id(next_question_id)
            if target_q:
                if self._question_queue is None:
                    # Convert to queue-based tracking first
                    self._question_queue = self._get_sorted_questions()[:]
                
                insert_at = self._queue_index + 1
                # Remove if already in queue to avoid duplicates
                target_id = self._normalize_id(next_question_id)
                self._question_queue = [
                    q for q in self._question_queue
                    if self._normalize_id(q.get("_id")) != target_id
                ]
                self._question_queue.insert(insert_at, target_q)
                self._queue_index += 1

                if end_after_branch:
                    # Truncate everything after the branched question so it is the last one shown
                    self._question_queue = self._question_queue[:self._queue_index + 1]
                    logger.info(f"Terminal branch: queue truncated to {len(self._question_queue)} questions")
            else:
                self._advance_index()
        else:
            self._advance_index()
        
        logger.info(
            f"Recorded answer for '{question_text}': '{answer}'. "
            f"Queue index now {self._queue_index} of {len(self._question_queue) if self._question_queue is not None else '?'}"
        )
        
        return self.get_current_question() is not None
    
    def _advance_index(self):
        """Move to next question sequentially"""
        if self._question_queue is not None:
            self._queue_index += 1
        else:
            self.current_question_index += 1
    
    def _resolve_next_question(self, answer: str, options: List[Dict[str, Any]]) -> tuple:
        """
        Given the user's answer and the question's options, return (next_question_id, end_after_branch).

        - next_question_id: the ID to branch to, or None
        - end_after_branch: True if the queue should be truncated after the branch (terminal option)

        Matches the selected option by numeric index (1-based) first, then by text (case-insensitive).
        This handles both WhatsApp (numbered replies) and chatbot (button text / typed text) inputs.
        """
        if not options:
            return (None, False)

        sorted_opts = sorted(options, key=lambda o: o.get("order", 0))
        answer_stripped = answer.strip()
        answer_lower = answer_stripped.lower()

        matched_opt = None

        # Try numeric match first (e.g. user typed "4" → pick 4th option)
        if answer_stripped.isdigit():
            number = int(answer_stripped)
            if 1 <= number <= len(sorted_opts):
                matched_opt = sorted_opts[number - 1]
                logger.info(f"Workflow: Numeric match #{number} -> option '{matched_opt.get('text')}'")

        # Fall back to exact text match (case-insensitive)
        if matched_opt is None:
            for opt in sorted_opts:
                opt_text = (opt.get("text") or "").strip()
                if opt_text.lower() == answer_lower:
                    matched_opt = opt
                    break

        # Optional: partial match for conversational/free-text answers
        if matched_opt is None and self._conversation_style_enabled():
            answer_tokens = set(re.findall(r"\w+", answer_lower))
            best_opt = None
            best_score = 0

            for opt in sorted_opts:
                opt_text = (opt.get("text") or "").strip()
                if not opt_text:
                    continue

                opt_lower = opt_text.lower()
                # Strong substring match first (e.g. "I prefer morning" -> "morning")
                if opt_lower and opt_lower in answer_lower:
                    matched_opt = opt
                    break

                opt_tokens = set(re.findall(r"\w+", opt_lower))
                # Keep only meaningful tokens to avoid matching on common short words
                opt_tokens = {t for t in opt_tokens if len(t) > 2}
                common = answer_tokens.intersection(opt_tokens)
                score = len(common)
                if score > best_score:
                    best_score = score
                    best_opt = opt

            if matched_opt is None and best_opt is not None and best_score > 0:
                matched_opt = best_opt

        if matched_opt is not None:
            is_terminal = bool(matched_opt.get("isTerminal"))
            next_id = matched_opt.get("nextQuestionId")

            if is_terminal and not next_id:
                # Terminal with no branch target — end the workflow immediately
                logger.info(f"Option '{matched_opt.get('text')}' is terminal with no next question, ending workflow")
                self._end_workflow()
                return (None, False)

            if is_terminal and next_id:
                # Terminal with a branch target — show that question last, then end
                logger.info(f"Option '{matched_opt.get('text')}' is terminal, will branch to '{next_id}' then end")
                return (next_id, True)

            return (next_id if next_id else None, False)

        return (None, False)
    
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
        target_id = self._normalize_id(question_id)
        questions = self.current_workflow.get("questions", [])
        for q in questions:
            if self._normalize_id(q.get("_id")) == target_id:
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
        self.current_service_name = None
        self.current_workflow = None
        self.current_question_index = 0
        self._question_queue = None
        self._queue_index = 0
        self.workflow_answers = {}
        self.is_active = False
        self._linked_question_ids = set()

    def export_state(self) -> Optional[Dict[str, Any]]:
        """JSON-friendly snapshot for web widget session resume."""
        if not self.is_active:
            return None
        q_copy = list(self._question_queue) if self._question_queue is not None else None
        return {
            "current_service_name": self.current_service_name,
            "current_workflow": self.current_workflow,
            "current_question_index": self.current_question_index,
            "_queue_index": self._queue_index,
            "_question_queue": q_copy,
            "workflow_answers": dict(self.workflow_answers),
            "_linked_question_ids": list(self._linked_question_ids),
            "is_active": self.is_active,
        }

    def import_state(self, data: Optional[Dict[str, Any]]) -> None:
        """Restore from export_state()."""
        self.reset()
        if not data or not isinstance(data, dict):
            return
        self.current_service_name = data.get("current_service_name")
        self.current_workflow = data.get("current_workflow")
        self.current_question_index = int(data.get("current_question_index", 0))
        self._queue_index = int(data.get("_queue_index", 0))
        self._question_queue = data.get("_question_queue")
        self.workflow_answers = dict(data.get("workflow_answers") or {})
        lids = data.get("_linked_question_ids")
        self._linked_question_ids = set(lids) if isinstance(lids, list) else set()
        self.is_active = bool(data.get("is_active"))
        if self.is_active and not self.current_workflow:
            self.is_active = False
