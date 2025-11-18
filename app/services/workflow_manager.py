"""Workflow manager to handle workflow questions in sequence"""
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger("assistly.workflow_manager")


class WorkflowManager:
    """Manages workflow question flow"""
    
    def __init__(self, context: Dict[str, Any]):
        self.context = context
        self.current_treatment_plan: Optional[str] = None
        self.current_workflow: Optional[Dict[str, Any]] = None
        self.current_question_index: int = 0
        self.workflow_answers: Dict[str, Any] = {}
        self.is_active: bool = False
    
    def start_workflow_for_treatment_plan(self, treatment_plan_name: str) -> bool:
        """Start workflow for a treatment plan. Returns True if workflow exists, False otherwise."""
        self.current_treatment_plan = treatment_plan_name
        
        # Find treatment plan in context (case-insensitive match)
        treatment_plans = self.context.get("treatment_plans", [])
        logger.info(f"Looking for treatment plan '{treatment_plan_name}' in {len(treatment_plans)} treatment plans")
        treatment_plan = None
        for tp in treatment_plans:
            if isinstance(tp, dict):
                tp_question = tp.get("question", "")
                # Case-insensitive comparison
                if tp_question.lower() == treatment_plan_name.lower():
                    treatment_plan = tp
                    logger.info(f"✓ Found treatment plan: '{tp_question}' with keys: {list(tp.keys())}")
                    break
        
        if not treatment_plan:
            logger.warning(f"✗ Treatment plan '{treatment_plan_name}' not found in context. Available plans: {[tp.get('question', '') if isinstance(tp, dict) else str(tp) for tp in treatment_plans[:5]]}")
            return False
        
        # Check for attached workflows
        attached_workflows = treatment_plan.get("attachedWorkflows", [])
        logger.info(f"Treatment plan has {len(attached_workflows)} attached workflows: {attached_workflows}")
        if not attached_workflows:
            logger.info(f"✗ No workflows attached to treatment plan '{treatment_plan_name}'")
            return False
        
        # Get the first workflow (assuming one workflow per treatment plan for now)
        workflow_id = attached_workflows[0].get("workflowId")
        logger.info(f"Looking for workflow with ID: '{workflow_id}'")
        if not workflow_id:
            logger.warning(f"✗ No workflowId in attachedWorkflows for '{treatment_plan_name}'. attachedWorkflows[0] = {attached_workflows[0]}")
            return False
        
        # Find workflow in context
        workflows = self.context.get("workflows", [])
        logger.info(f"Searching for workflow in {len(workflows)} workflows in context")
        workflow = None
        for wf in workflows:
            wf_id = wf.get("_id")
            if wf_id == workflow_id:
                workflow = wf
                logger.info(f"✓ Found workflow '{workflow_id}' with {len(wf.get('questions', []))} questions")
                break
        
        if not workflow:
            logger.warning(f"✗ Workflow '{workflow_id}' not found in context. Available workflow IDs: {[wf.get('_id') for wf in workflows[:5]]}")
            return False
        
        # Get questions from workflow, sorted by order
        questions = workflow.get("questions", [])
        if not questions:
            logger.info(f"Workflow '{workflow_id}' has no questions")
            return False
        
        # Sort questions by order
        sorted_questions = sorted(
            [q for q in questions if q.get("isActive", True)],
            key=lambda q: q.get("order", 0)
        )
        
        if not sorted_questions:
            logger.info(f"Workflow '{workflow_id}' has no active questions")
            return False
        
        self.current_workflow = workflow
        self.current_question_index = 0
        self.workflow_answers = {}
        self.is_active = True
        
        logger.info(
            f"Started workflow '{workflow_id}' for treatment plan '{treatment_plan_name}' "
            f"with {len(sorted_questions)} questions"
        )
        return True
    
    def get_current_question(self) -> Optional[Dict[str, Any]]:
        """Get the current question to ask"""
        if not self.is_active or not self.current_workflow:
            return None
        
        questions = self.current_workflow.get("questions", [])
        sorted_questions = sorted(
            [q for q in questions if q.get("isActive", True)],
            key=lambda q: q.get("order", 0)
        )
        
        if self.current_question_index >= len(sorted_questions):
            return None
        
        return sorted_questions[self.current_question_index]
    
    def record_answer(self, answer: str) -> bool:
        """Record answer to current question and move to next. Returns True if more questions remain."""
        if not self.is_active:
            return False
        
        current_question = self.get_current_question()
        if not current_question:
            return False
        
        question_id = current_question.get("_id")
        question_text = current_question.get("question", "")
        
        # Store answer
        self.workflow_answers[question_id] = {
            "question": question_text,
            "answer": answer,
            "order": current_question.get("order", 0)
        }
        
        # Move to next question
        self.current_question_index += 1
        
        logger.info(
            f"Recorded answer for question '{question_text}': '{answer}'. "
            f"Question {self.current_question_index} of {self._get_total_questions()}"
        )
        
        # Check if more questions remain
        return self.get_current_question() is not None
    
    def _get_total_questions(self) -> int:
        """Get total number of questions in current workflow"""
        if not self.current_workflow:
            return 0
        questions = self.current_workflow.get("questions", [])
        return len([q for q in questions if q.get("isActive", True)])
    
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
        self.workflow_answers = {}
        self.is_active = False

