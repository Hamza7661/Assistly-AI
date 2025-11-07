from typing import Any, Dict, List, Optional
import logging
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

logger = logging.getLogger("assistly.rag")


class RAGService:
    """Retrieval-Augmented Generation service using LangChain"""
    
    def __init__(self, settings: Any) -> None:
        self.openai_api_key: Optional[str] = settings.openai_api_key
        self.gpt_model: str = getattr(settings, 'gpt_model', 'gpt-4o-mini')
        self.rag_k: int = getattr(settings, 'rag_k', 3)
        self.rag_persist_directory: Optional[str] = getattr(settings, 'rag_persist_directory', None)
        self.embeddings = None
        self.llm = None
        self.vector_store: Optional[Chroma] = None
        self.retriever = None
        self.qa_chain = None
        self.qa_prompt = None
        self._retriever_method = None  # Cache the correct method to use
        
        # RAG is always enabled - initialize embeddings and LLM if API key is available
        if self.openai_api_key:
            try:
                # Initialize embeddings
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=self.openai_api_key,
                    model="text-embedding-3-small"
                )
                # Initialize LLM for QA chain
                self.llm = ChatOpenAI(
                    openai_api_key=self.openai_api_key,
                    model_name=self.gpt_model,
                    temperature=0.3
                )
                logger.info("RAG service initialized with OpenAI embeddings and LLM")
            except Exception as e:
                logger.error(f"Failed to initialize RAG components: {e}")
        else:
            logger.warning("OpenAI API key not provided, RAG will not be available")
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve documents using the retriever - handles API compatibility"""
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        
        # Determine which method to use on first call and cache it
        if self._retriever_method is None:
            if hasattr(self.retriever, 'invoke'):
                self._retriever_method = 'invoke'
            elif hasattr(self.retriever, 'get_relevant_documents'):
                self._retriever_method = 'get_relevant_documents'
            else:
                raise ValueError("Retriever doesn't have invoke() or get_relevant_documents() method")
        
        # Use the cached method
        if self._retriever_method == 'invoke':
            return self.retriever.invoke(query)
        else:
            return self.retriever.get_relevant_documents(query)
    
    def _prepare_documents_from_context(self, context: Dict[str, Any]) -> List[Document]:
        """Convert ALL context data into LangChain Documents for comprehensive RAG"""
        documents = []
        
        # Add Lead Types - critical for matching user responses
        lead_types = context.get("lead_types", [])
        if isinstance(lead_types, list):
            for i, lead_type in enumerate(lead_types):
                if isinstance(lead_type, dict):
                    value = lead_type.get("value", "")
                    text = lead_type.get("text", "")
                    if value and text:
                        doc_text = f"Lead Type Option {i+1}:\nValue: {value}\nText: {text}\nDescription: This is a lead type option that users can select."
                        documents.append(Document(
                            page_content=doc_text,
                            metadata={"source": "lead_type", "index": i, "type": "lead_type", "value": value, "text": text}
                        ))
        
        # Add Service Types - critical for matching user responses
        services = context.get("service_types", [])
        service_index = 0
        if isinstance(services, list):
            for service in services:
                if isinstance(service, dict):
                    name = service.get("name", service.get("title", ""))
                    description = service.get("description", "")
                    if name:
                        service_index += 1
                        doc_text = f"Service Option {service_index}:\nName: {name}"
                        if description:
                            doc_text += f"\nDescription: {description}"
                        doc_text += "\nThis is a service option that users can select."
                        documents.append(Document(
                            page_content=doc_text,
                            metadata={"source": "service", "index": service_index, "type": "service", "name": name}
                        ))
        
        # Add Treatment Plans - index both as services AND as Q&A for answering questions
        treatment_plans = context.get("treatment_plans", [])
        if isinstance(treatment_plans, list):
            for i, plan in enumerate(treatment_plans):
                if isinstance(plan, dict):
                    question = plan.get("question", "")
                    answer = plan.get("answer", "")
                    description = plan.get("description", "")
                    
                    if question:
                        service_index += 1
                        # Index as service option (for selection)
                        doc_text = f"Service Option {service_index}:\nName: {question}"
                        if description:
                            doc_text += f"\nDescription: {description}"
                        doc_text += "\nThis is a service option that users can select (treatment plan)."
                        documents.append(Document(
                            page_content=doc_text,
                            metadata={"source": "service", "index": service_index, "type": "service", "name": question, "is_treatment_plan": True}
                        ))
                        
                        # Also index as Q&A document (for answering questions about this treatment)
                        if answer:
                            qa_text = f"Treatment Plan {i+1}:\nQuestion: {question}\nAnswer: {answer}"
                            if description:
                                qa_text += f"\nDescription: {description}"
                            documents.append(Document(
                                page_content=qa_text,
                                metadata={"source": "treatment_plan", "index": i, "type": "treatment_plan_qa", "question": question, "has_answer": True}
                            ))
                        elif description:
                            # If no answer but has description, use description as answer
                            qa_text = f"Treatment Plan {i+1}:\nQuestion: {question}\nAnswer: {description}"
                            documents.append(Document(
                                page_content=qa_text,
                                metadata={"source": "treatment_plan", "index": i, "type": "treatment_plan_qa", "question": question, "has_answer": True}
                            ))
        
        # Add FAQs
        faqs = context.get("faqs", [])
        if isinstance(faqs, list):
            for i, faq in enumerate(faqs):
                if isinstance(faq, dict):
                    question = faq.get("question", "")
                    answer = faq.get("answer", faq.get("response", ""))
                    if question and answer:
                        doc_text = f"FAQ {i+1}:\nQuestion: {question}\nAnswer: {answer}"
                        documents.append(Document(
                            page_content=doc_text,
                            metadata={"source": "faq", "index": i, "type": "faq"}
                        ))
                elif isinstance(faq, str):
                    documents.append(Document(
                        page_content=f"FAQ {i+1}: {faq}",
                        metadata={"source": "faq", "index": i, "type": "faq"}
                    ))
        
        # Add profession description
        profession = context.get("profession", "")
        if profession:
            documents.append(Document(
                page_content=f"About our {profession}: {profession}\nThis is the type of business/profession.",
                metadata={"source": "profession", "type": "profession"}
            ))
        
        # Add integration info (greeting, assistant name, validation flags, etc.)
        integration = context.get("integration", {})
        if isinstance(integration, dict):
            assistant_name = integration.get("assistantName", "")
            greeting = integration.get("greeting", "")
            # Default greeting if null/empty
            if not greeting or not str(greeting).strip():
                greeting = "Hi! How can I help you today?"
            else:
                greeting = str(greeting).strip()
            
            validate_email = integration.get("validateEmail", True)
            validate_phone = integration.get("validatePhoneNumber", True)
            
            doc_text = ""
            if assistant_name:
                doc_text += f"Assistant Name: {assistant_name}\n"
            doc_text += f"Greeting Message: {greeting}\n"
            doc_text += f"Email Validation: {'Enabled' if validate_email else 'Disabled'}\n"
            doc_text += f"Phone Validation: {'Enabled' if validate_phone else 'Disabled'}"
            
            if doc_text.strip():
                documents.append(Document(
                    page_content=doc_text.strip(),
                    metadata={"source": "integration", "type": "integration", "validateEmail": validate_email, "validatePhoneNumber": validate_phone, "greeting": greeting}
                ))
        
        logger.info(f"Prepared {len(documents)} documents from context for RAG (including lead types, service types, treatment plans with answers, FAQs, and validation flags)")
        return documents
    
    def build_vector_store(self, context: Dict[str, Any], persist_directory: Optional[str] = None) -> bool:
        """Build vector store from context data"""
        if not self.embeddings:
            logger.warning("Embeddings not initialized, skipping vector store creation")
            return False
        
        try:
            # Prepare documents
            documents = self._prepare_documents_from_context(context)
            
            if not documents:
                logger.warning("No documents to index for RAG")
                return False
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            splits = text_splitter.split_documents(documents)
            
            logger.info(f"Split {len(documents)} documents into {len(splits)} chunks")
            
            # Use persist_directory from settings if not provided
            persist_dir = persist_directory or self.rag_persist_directory
            
            # Create vector store (in-memory by default, or persistent if directory provided)
            if persist_dir:
                self.vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory=persist_dir
                )
                logger.info(f"Created persistent vector store at {persist_dir}")
            else:
                self.vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings
                )
                logger.info("Created in-memory vector store")
            
            # Create retriever with configured k value
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.rag_k}  # Use configured k value
            )
            
            # Create QA chain with strict prompt to ensure accurate responses
            if self.llm:
                self._create_qa_chain()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build vector store: {e}")
            return False
    
    def _create_qa_chain(self) -> None:
        """Create a comprehensive QA chain for ALL conversation responses"""
        # Note: Flow will be adjusted dynamically based on is_whatsapp parameter
        prompt_template = """You are a {profession} assistant. Use ONLY the context below.

RULES:
1. Match user input to lead types, services, FAQs from context - use exact values
2. Use the greeting from context ONLY for the very first message - do NOT repeat it
3. {flow_instruction}
4. OPTION FORMAT: {option_format}
5. CONVERSATION FLOW TRACKING (CRITICAL):
   - Analyze conversation history CAREFULLY to determine what information has been collected
   - Extract collected info: leadType, serviceType, leadName, leadEmail, leadPhoneNumber, title (title is the text from the selected lead type)
   - Flow progression (STRICT ORDER - DO NOT SKIP STEPS):
     * Step 1: If NO lead type collected → ask for lead type and show ALL lead types using {option_format}
     * Step 2: If lead type collected but NO service type → MANDATORY: Ask "Which service are you interested in?" and show ALL services from context using {option_format} (ALL services must be shown)
       ** CRITICAL: Service type is REQUIRED for ALL lead types including "callback", "appointment arrangement", "further information" - DO NOT SKIP THIS STEP
       ** You MUST ask for service type immediately after lead type is selected, even for callback requests
       ** NEVER go directly from lead type to name - service type is ALWAYS required
     * Step 3: If service type collected but NO name → ask for name
     * Step 4: If name collected but NO email → ask for email
     * Step 5: If email collected but NO phone (and not WhatsApp) → ask for phone
     * Step 6: If all info collected → generate JSON immediately
   - NEVER skip Step 2 (service type) - it is MANDATORY for every conversation
   - CRITICAL: When showing service options, you MUST show ALL services listed in the context, not just one or a few
   - For WhatsApp: Use numbered list format (1. Option 1, 2. Option 2, etc.)
   - For Web: Use button format (<button> Option Text </button> or <button value="value"> Text </button>)
   - Do NOT repeat questions already asked
   - Do NOT restart the conversation - continue from where you left off
   - If user provides information (name, email, phone), acknowledge and move to next step
5. OTP HANDLING (Use semantic understanding - detect user intent):
   - If user wants to resend OTP to the SAME contact (lost code, didn't receive, send again, resend, didn't get it, can't find code) → respond with ONLY: RETRY_OTP_REQUESTED
   - If user wants to CHANGE phone number (wrong number, different phone, send to another phone, new phone, that's not my number) → respond with ONLY: CHANGE_PHONE_REQUESTED
   - If user wants to CHANGE email address (wrong email, different email, send to another email, new email, that's not my email) → respond with ONLY: CHANGE_EMAIL_REQUESTED
   - Use semantic understanding to detect intent - analyze what the user means, not exact words
   - These special responses are REQUIRED - do NOT add any other text
6. VALIDATION: Check context for validateEmail and validatePhoneNumber flags - only validate if enabled
7. When all info collected, output ONLY JSON: {json_fields}
8. Be conversational and helpful

{conversation_history}

Context (lead types, services, FAQs, greeting, validation flags):
{context}

User: {question}

Response:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "profession", "flow_instruction", "json_fields", "conversation_history", "option_format"]
        )
        
        try:
            # Try to create document chain (newer LangChain API)
            document_chain = create_stuff_documents_chain(self.llm, PROMPT)
            self.qa_chain = document_chain
            logger.info("QA chain created with document chain (newer API)")
        except Exception as e:
            logger.warning(f"Could not create document chain, will use direct LLM calls: {e}")
            # Fallback: store prompt template for direct use
            self.qa_chain = None
            self.qa_prompt = PROMPT
    
    async def get_relevant_context(self, query: str, k: Optional[int] = None) -> str:
        """Retrieve relevant context for a user query with enhanced formatting"""
        if not self.retriever:
            logger.warning("Retriever not initialized, returning empty context")
            return ""
        
        try:
            # Retrieve relevant documents
            docs = self._retrieve_documents(query)
            
            if not docs:
                logger.info(f"No relevant documents found for query: {query}")
                return ""
            
            # Enhanced formatting with source information
            context_parts = []
            for i, doc in enumerate(docs, 1):
                doc_type = doc.metadata.get("type", "unknown")
                source = doc.metadata.get("source", "knowledge_base")
                page_content = doc.page_content
                
                # Format with clear source attribution
                context_parts.append(
                    f"[SOURCE: {source.upper()} | TYPE: {doc_type.upper()}]\n"
                    f"{page_content}"
                )
            
            context = "\n\n---\n\n".join(context_parts)
            logger.info(f"Retrieved {len(docs)} relevant documents for query: {query[:50]}...")
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving relevant context: {e}")
            return ""
    
    def _extract_collected_info(self, conversation_history: Optional[List[Dict[str, str]]], context_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Explicitly extract what information has been collected from conversation history"""
        collected = {
            "leadType": None,
            "serviceType": None,
            "leadName": None,
            "leadEmail": None,
            "leadPhoneNumber": None
        }
        
        if not conversation_history or not context_data:
            return collected
        
        # Get available options
        lead_types = context_data.get("lead_types", [])
        services = context_data.get("service_types", [])
        treatment_plans = context_data.get("treatment_plans", [])
        
        # Combine services and treatment plans (services may already be merged)
        all_services = []
        seen_services = set()
        
        # Services can be strings or dicts (after merging, treatment plans are dicts)
        for s in services:
            if isinstance(s, dict):
                service_name = s.get("name", s.get("title", ""))
            else:
                service_name = str(s)
            
            if service_name and service_name.lower() not in seen_services:
                all_services.append(service_name)
                seen_services.add(service_name.lower())
        
        # Also check treatment_plans in case they weren't merged yet
        for tp in treatment_plans:
            if isinstance(tp, dict):
                service_name = tp.get("question", "")
                if service_name and service_name.lower() not in seen_services:
                    all_services.append(service_name)
                    seen_services.add(service_name.lower())
        
        # Extract lead type values
        lead_type_values = []
        for lt in lead_types:
            if isinstance(lt, dict):
                lead_type_values.append(lt.get("value", "").lower())
                lead_type_values.append(lt.get("text", "").lower())
        
        # Scan conversation history (most recent first to get latest selections)
        for msg in reversed(conversation_history):
            content = msg.get("content", "")
            content_lower = content.lower()
            role = msg.get("role", "")
            
            if role == "user":
                # Check for lead type (only if not already collected)
                if not collected["leadType"]:
                    for lt_val in lead_type_values:
                        if lt_val and lt_val in content_lower:
                            collected["leadType"] = lt_val
                            break
                
                # Check for service type (only if lead type is collected and service not yet collected)
                if collected["leadType"] and not collected["serviceType"]:
                    # Match against all services (case-insensitive, exact or partial match)
                    user_message_clean = content_lower.strip()
                    for service in all_services:
                        if service:
                            service_lower = service.lower().strip()
                            # Check for exact match first (most reliable)
                            if service_lower == user_message_clean:
                                collected["serviceType"] = service  # Use original case
                                break
                            # Then check if service name is contained in user message (but not too short)
                            elif len(service_lower) >= 3 and service_lower in user_message_clean:
                                collected["serviceType"] = service  # Use original case
                                break
                
                # Check for name (simple heuristic: if assistant asked for name and user replied)
                if "name" in content and len(content.split()) < 5:
                    # Likely a name
                    collected["leadName"] = content.strip()
                
                # Check for email
                if "@" in content or "email" in content:
                    # Extract email
                    import re
                    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', msg.get("content", ""))
                    if email_match:
                        collected["leadEmail"] = email_match.group()
                
                # Check for phone
                if any(char.isdigit() for char in content) and len([c for c in content if c.isdigit()]) >= 10:
                    import re
                    phone_match = re.search(r'[\d\s\+\-\(\)]{10,}', msg.get("content", ""))
                    if phone_match:
                        collected["leadPhoneNumber"] = phone_match.group().strip()
        
        return collected
    
    def _merge_treatment_plans_into_services(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Merge treatment plans into service_types array for unified service selection (same as gpt_service)"""
        service_types = context.get("service_types", [])
        treatment_plans = context.get("treatment_plans", [])
        
        # Convert treatment plans to service format and merge with service types
        merged_services = list(service_types)  # Start with existing services (can be strings or dicts)
        for plan in treatment_plans:
            if isinstance(plan, dict) and "question" in plan:
                # Convert treatment plan to service format
                service_item = {
                    "name": plan["question"],
                    "title": plan["question"],
                    "description": plan.get("description", ""),
                    "is_treatment_plan": True
                }
                merged_services.append(service_item)
        
        # Update context with merged services
        context = context.copy()
        context["service_types"] = merged_services
        return context
    
    async def get_accurate_answer(self, query: str, profession: str = "Clinic", is_whatsapp: bool = False, context_data: Optional[Dict[str, Any]] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
        """Get an accurate answer using LangChain - let AI handle flow progression and JSON generation"""
        if not self.llm or not self.retriever:
            logger.warning("LLM or retriever not initialized")
            return None
        
        try:
            # Merge treatment plans into service types (same as gpt_service does)
            if context_data:
                context_data = self._merge_treatment_plans_into_services(context_data)
            
            # Retrieve relevant documents for context
            docs = self._retrieve_documents(query)
            
            if not docs:
                logger.info(f"No relevant documents found for query: {query}")
                return None
            
            # Format context from documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Get validation flags - CRITICAL for LangChain to know when to send OTP
            validate_email = True
            validate_phone = True
            if context_data:
                integration = context_data.get("integration", {})
                validate_email = integration.get("validateEmail", True)
                validate_phone = integration.get("validatePhoneNumber", True)
            
            # Check OTP verification status from conversation history
            email_otp_sent = False
            email_otp_verified = False
            phone_otp_sent = False
            phone_otp_verified = False
            
            if conversation_history:
                for msg in reversed(conversation_history):
                    content = msg.get("content", "").lower()
                    if "sent a 6-digit verification code" in content and "email" in content:
                        email_otp_sent = True
                    if "email has been verified" in content or "email verified" in content:
                        email_otp_verified = True
                    if "sent a 6-digit verification code" in content and "phone" in content:
                        phone_otp_sent = True
                    if "phone has been verified" in content or "phone verified" in content:
                        phone_otp_verified = True
            
            # Format validation status strings for prompt
            email_validation_status = f"OTP sent: {email_otp_sent}, Verified: {email_otp_verified}" if validate_email else "Not required"
            phone_validation_status = f"OTP sent: {phone_otp_sent}, Verified: {phone_otp_verified}" if validate_phone and not is_whatsapp else "Not required"
            email_validation_enabled = "ENABLED" if validate_email else "DISABLED"
            phone_validation_enabled = "ENABLED" if validate_phone else "DISABLED"
            
            # Prepare all available options for LangChain
            lead_types = context_data.get("lead_types", []) if context_data else []
            services = context_data.get("service_types", []) if context_data else []
            
            lead_types_text = "\n".join([f"- {lt.get('text', '')} (value: {lt.get('value', '')})" for lt in lead_types if isinstance(lt, dict)])
            
            all_services = []
            for s in services:
                if isinstance(s, dict):
                    all_services.append(s.get("name", s.get("title", "")))
                else:
                    all_services.append(str(s))
            services_text = "\n".join([f"- {s}" for s in all_services])
            
            # Format conversation history - let AI analyze what's been collected
            # Include FULL history from the start (greeting, lead type selection, etc.)
            history_text = ""
            if conversation_history:
                history_text = "\n\nCONVERSATION HISTORY (FULL - FROM START):\n"
                for msg in conversation_history:  # Include ALL messages from the beginning
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role and content:
                        history_text += f"{role.upper()}: {content}\n"
            
            # Determine flow and JSON fields - include OTP verification steps if validation is enabled
            # Note: For WhatsApp, phone is already verified, so skip phone OTP steps even if phone validation is enabled
            if is_whatsapp:
                if validate_email:
                    flow = "lead type → service type → name → email → send email OTP → verify email OTP → JSON (phone from WhatsApp, already verified)"
                else:
                    flow = "lead type → service type → name → email → JSON (phone from WhatsApp, already verified)"
                json_fields = '{"leadType": "...", "serviceType": "...", "leadName": "...", "leadEmail": "...", "title": "..."}'
                option_format = "NUMBERED LIST (1. Option 1, 2. Option 2, etc.)"
            else:
                # For web chat, include phone OTP steps if phone validation is enabled
                if validate_email and validate_phone:
                    flow = "lead type → service type → name → email → send email OTP → verify email OTP → phone → send phone OTP → verify phone OTP → JSON"
                elif validate_email:
                    flow = "lead type → service type → name → email → send email OTP → verify email OTP → phone → JSON"
                elif validate_phone:
                    flow = "lead type → service type → name → email → phone → send phone OTP → verify phone OTP → JSON"
                else:
                    flow = "lead type → service type → name → email → phone → JSON"
                json_fields = '{"leadType": "...", "serviceType": "...", "leadName": "...", "leadEmail": "...", "leadPhoneNumber": "...", "title": "..."}'
                option_format = "BUTTONS: Use <button> Option Text </button> format for all options"
            
            # Let LangChain handle everything - flow progression, matching, and JSON generation
            prompt = f"""You are a {profession} assistant. Follow this conversation flow: {flow}

AVAILABLE OPTIONS:
Lead Types:
{lead_types_text}

Services:
{services_text}

RULES:
1. CRITICAL: Analyze the conversation history below CAREFULLY to determine what information has already been collected (lead type, service type, name, email, phone, title) - do NOT ask for information that's already collected
   - "title" is the text from the selected lead type (e.g., if user selected "I would like to arrange an appointment", title = "I would like to arrange an appointment")
2. Match user input to exact values from the options above
3. Follow the flow strictly: {flow}
   - IMPORTANT: OTP verification steps (send OTP → verify OTP) are MANDATORY if validation is enabled - do NOT skip them
   - Do NOT generate JSON until ALL OTP verification steps are complete (if validation is enabled)
4. When showing options, use {option_format} and show ALL options for the CURRENT step only (lead types OR services, not both)
   - CRITICAL: For buttons, use EXACT format: <button>Option Text</button> with NO spaces inside angle brackets
   - WRONG: < button >Text< /button > or <button >Text</ button>
   - CORRECT: <button>Text</button>
5. OTP HANDLING (HIGHEST PRIORITY - CHECK THIS FIRST BEFORE ANY OTHER RESPONSE):
   - VALIDATION REQUIREMENTS: Email validation is {email_validation_enabled}. Phone validation is {phone_validation_enabled}.
   - FORBIDDEN: NEVER use SEND_EMAIL or SEND_PHONE formats unless user ACTUALLY provides email/phone in their CURRENT message
   - FORBIDDEN: NEVER use SEND_EMAIL or SEND_PHONE formats for asking questions - use natural language instead
   - CRITICAL: If email validation is ENABLED and user provides email in CURRENT message - respond with: "SEND_EMAIL: [email]" where [email] is the actual email from user's message
   - CRITICAL: If email validation is DISABLED and user provides email - acknowledge and ask for phone number using natural language (e.g., "Please provide your phone number") - do NOT use SEND_PHONE format
   - CRITICAL: If phone validation is ENABLED and NOT WhatsApp and user provides phone in CURRENT message - respond with: "SEND_PHONE: [phone]" where [phone] is the actual phone number from user's message
   - CRITICAL: If phone validation is DISABLED and user provides phone - acknowledge and generate JSON (if all info collected) - do NOT use SEND_PHONE format
   - CRITICAL: If conversation history shows OTP was sent to an email/phone, and user provides a DIFFERENT email/phone, this is ALWAYS a CHANGE request
   - If conversation history shows "I've sent a 6-digit verification code to [email]" and user mentions a different email → respond with: "CHANGE_EMAIL_REQUESTED: [new_email]" where [new_email] is the new email from user's message
   - If conversation history shows "I've sent a 6-digit verification code to [phone]" and user mentions a different phone → respond with: "CHANGE_PHONE_REQUESTED: [new_phone]" where [new_phone] is the new phone from user's message
   - Examples of email change requests: "wrong one", "send it to [new email]", "change email to [new email]", "use [new email] instead", "sorry send it to [new email]", "oh no wrong one! plz send it to [new email]", "that's not my email, send to [new email]"
   - Examples of phone change requests: "wrong number", "send it to [new phone]", "change phone to [new phone]", "use [new phone] instead", "that's not my number"
   - If user wants to resend OTP to same contact → respond with ONLY: RETRY_OTP_REQUESTED
   - FORMAT: Always include the email/phone in your response when sending OTP or changing contact:
     * For sending email OTP: "SEND_EMAIL: [email]"
     * For sending phone OTP: "SEND_PHONE: [phone]"
     * For changing email: "CHANGE_EMAIL_REQUESTED: [new_email]"
     * For changing phone: "CHANGE_PHONE_REQUESTED: [new_phone]"
   - IMPORTANT: If OTP verification is in progress (OTP sent but not verified), do NOT generate buttons, do NOT generate JSON - only respond with CHANGE_EMAIL_REQUESTED, CHANGE_PHONE_REQUESTED, or RETRY_OTP_REQUESTED
   - CRITICAL: You must detect email/phone from user's CURRENT message ONLY - do NOT extract from previous messages or conversation history
   - CRITICAL: If user provides email but NO phone number in their CURRENT message, you MUST ask for phone number first (if not WhatsApp) - do NOT try to send phone OTP or extract phone from conversation history
   - CRITICAL: When asking for phone number, use NATURAL LANGUAGE like "Please provide your phone number" - do NOT use SEND_PHONE format unless user actually provides a phone number
   - CRITICAL: Only send OTP when the user ACTUALLY provides the email/phone in their CURRENT message - do NOT assume or extract from previous messages
   - CRITICAL: SEND_PHONE format is ONLY for when user provides an actual phone number - do NOT use it for asking questions or placeholders
6. JSON GENERATION (ONLY when validation is complete):
   - Email validation status: {email_validation_status}
   - Phone validation status: {phone_validation_status}
   - CRITICAL: Do NOT generate JSON if email validation is ENABLED but email OTP is not verified (current: {email_otp_verified if validate_email else "N/A"})
   - CRITICAL: Do NOT generate JSON if phone validation is ENABLED (and not WhatsApp) but phone OTP is not verified (current: {phone_otp_verified if validate_phone and not is_whatsapp else "N/A"}). Dont assume it is verified see the history to make your judgment
   - When ALL required information is collected AND no OTP change requests detected AND OTP verification is complete (if required), output ONLY valid JSON: {json_fields}
7. Do NOT repeat questions already answered - continue from where conversation left off
8. Move to next step automatically when information is collected

{history_text}

Context:
{context}

User: {query}

Response:"""
            
            response = await self.llm.ainvoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Check if response is JSON (all info collected)
            answer_stripped = answer.strip()
            if answer_stripped.startswith("{") and answer_stripped.endswith("}"):
                try:
                    # Validate it's proper JSON
                    json.loads(answer_stripped)
                    logger.info("LangChain generated JSON - all info collected")
                    return answer_stripped
                except:
                    pass
            
            # Return LangChain's response as-is (AI handles everything)
            return answer.strip() if answer else None
            
        except Exception as e:
            logger.error(f"Error getting accurate answer: {e}")
            return None
    
    async def _check_otp_requests(self, query: str) -> Optional[str]:
        """Use LangChain to semantically detect OTP retry/change requests"""
        query_lower = query.lower()
        
        # Retry patterns
        retry_patterns = ["resend", "send again", "didn't receive", "didn't get", "lost code", "can't find", "haven't received"]
        if any(pattern in query_lower for pattern in retry_patterns):
            return "RETRY_OTP_REQUESTED"
        
        # Change phone patterns
        change_phone_patterns = ["wrong number", "different phone", "another phone", "new phone", "change phone", "not my number"]
        if any(pattern in query_lower for pattern in change_phone_patterns):
            return "CHANGE_PHONE_REQUESTED"
        
        # Change email patterns
        change_email_patterns = ["wrong email", "different email", "another email", "new email", "change email", "not my email"]
        if any(pattern in query_lower for pattern in change_email_patterns):
            return "CHANGE_EMAIL_REQUESTED"
        
        return None
    
    async def _generate_step_response(self, current_step: str, query: str, collected: Dict[str, Any], context_data: Optional[Dict[str, Any]], profession: str, is_whatsapp: bool) -> Optional[str]:
        """Generate response for current step using LangChain for matching and natural language"""
        if not self.llm or not context_data:
            return None
        
        option_format = "NUMBERED LIST (1. Option 1, 2. Option 2, etc.)" if is_whatsapp else "BUTTONS (<button> Option Text </button>)"
        
        if current_step == "ask_lead_type":
            # Use LangChain to match user input to lead types
            lead_types = context_data.get("lead_types", [])
            lead_type_text = "\n".join([f"- {lt.get('text', '')}" for lt in lead_types if isinstance(lt, dict)])
            
            prompt = f"""You are a {profession} assistant. The user said: "{query}"

Match their input to one of these lead types:
{lead_type_text}

If it matches a lead type, acknowledge it. If not, ask them to choose from the options above using {option_format}.

Response:"""
            
            response = await self.llm.ainvoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Add buttons/numbers
            if is_whatsapp:
                numbered = "\n".join([f"{i}. {lt.get('text', '')}" for i, lt in enumerate(lead_types, 1) if isinstance(lt, dict)])
                return f"{answer}\n\n{numbered}"
            else:
                buttons = "\n".join([f"<button value=\"{lt.get('value', '')}\">{lt.get('text', '')}</button>" for lt in lead_types if isinstance(lt, dict)])
                return f"{answer}\n\n{buttons}"
        
        elif current_step == "ask_service_type":
            # Merge treatment plans into service types (same as gpt_service does)
            services = context_data.get("service_types", [])
            treatment_plans = context_data.get("treatment_plans", [])
            
            # Start with regular services
            all_services = []
            seen_services = set()  # Track to avoid duplicates
            
            # Add regular services (can be strings or dicts)
            for s in services:
                if isinstance(s, dict):
                    service_name = s.get("name", s.get("title", ""))
                else:
                    service_name = str(s)
                
                if service_name and service_name not in seen_services:
                    all_services.append(service_name)
                    seen_services.add(service_name)
            
            # Add treatment plans as services
            for tp in treatment_plans:
                if isinstance(tp, dict):
                    service_name = tp.get("question", "")
                    if service_name and service_name not in seen_services:
                        all_services.append(service_name)
                        seen_services.add(service_name)
            
            if not all_services:
                return "Which service are you interested in?"
            
            service_text = "\n".join([f"- {s}" for s in all_services])
            
            prompt = f"""You are a {profession} assistant. The user said: "{query}"

Available services:
{service_text}

If their input matches one of these services, acknowledge it briefly. Otherwise, ask "Which service are you interested in?" in a friendly way.

IMPORTANT: Do NOT list all services in your response - just acknowledge or ask. All services will be shown as buttons/numbers separately.

Response (just the text, no buttons/numbers - they will be added separately):"""
            
            response = await self.llm.ainvoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Strip any existing buttons/numbers from LangChain response
            import re
            answer = re.sub(r'<button[^>]*>.*?</button>', '', answer, flags=re.DOTALL)
            answer = re.sub(r'\d+\.\s+[^\n]+', '', answer)  # Remove numbered lists
            answer = answer.strip()
            
            # Add buttons/numbers - ensure no duplicates
            if is_whatsapp:
                numbered = "\n".join([f"{i}. {s}" for i, s in enumerate(all_services, 1)])
                return f"{answer}\n\n{numbered}" if answer else numbered
            else:
                buttons = "\n".join([f"<button>{s}</button>" for s in all_services])
                return f"{answer}\n\n{buttons}" if answer else buttons
        
        elif current_step == "ask_name":
            prompt = f"""You are a {profession} assistant. The user said: "{query}"

Extract their name from their response. If it's a name, acknowledge it warmly and ask for their email. If not clear, ask "What's your name?" in a friendly way.

Response:"""
            response = await self.llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        
        elif current_step == "ask_email":
            prompt = f"""You are a {profession} assistant. The user said: "{query}"

Extract their email address. If you found an email, acknowledge it. If not, ask "Could you please provide your email address?" in a friendly way.

Response:"""
            response = await self.llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        
        elif current_step == "ask_phone":
            prompt = f"""You are a {profession} assistant. The user said: "{query}"

Extract their phone number. If you found a phone number, acknowledge it. If not, ask "What's your phone number?" in a friendly way.

Response:"""
            response = await self.llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        
        elif current_step == "generate_json":
            # Generate JSON with collected data
            if is_whatsapp:
                json_data = {
                    "leadType": collected.get("leadType", ""),
                    "serviceType": collected.get("serviceType", ""),
                    "leadName": collected.get("leadName", ""),
                    "leadEmail": collected.get("leadEmail", "")
                }
            else:
                json_data = {
                    "leadType": collected.get("leadType", ""),
                    "serviceType": collected.get("serviceType", ""),
                    "leadName": collected.get("leadName", ""),
                    "leadEmail": collected.get("leadEmail", ""),
                    "leadPhoneNumber": collected.get("leadPhoneNumber", "")
                }
            return json.dumps(json_data)
        
        return None
    
    async def _get_direct_answer(self, query: str, context: str, profession: str, is_whatsapp: bool = False, validate_email: bool = True, validate_phone: bool = True, conversation_history: str = "") -> str:
        """Get answer directly from LLM using prompt template"""
        # Different flow for WhatsApp (no phone collection)
        if is_whatsapp:
            flow_instruction = "Follow STRICT flow order: lead type → service type (MANDATORY) → name → email → JSON (phone already available from WhatsApp). Service type is REQUIRED for ALL lead types including callback - NEVER skip it."
            json_fields = '{"leadType": "...", "serviceType": "...", "leadName": "...", "leadEmail": "...", "title": "..."}'
            option_format = "NUMBERED LIST (WhatsApp): Show options as numbered list: 1. Option 1, 2. Option 2, 3. Option 3, etc."
        else:
            flow_instruction = "Follow STRICT flow order: lead type → service type (MANDATORY) → name → email → phone → JSON. Service type is REQUIRED for ALL lead types including callback - NEVER skip it."
            json_fields = '{"leadType": "...", "serviceType": "...", "leadName": "...", "leadEmail": "...", "leadPhoneNumber": "...", "title": "..."}'
            option_format = "BUTTONS (Web): Show options as buttons: <button> Option Text </button> or <button value=\"value\"> Text </button>"
        
        # Add validation info
        validation_note = ""
        if not validate_email:
            validation_note += " Email validation is DISABLED - do not send email OTP. "
        if not validate_phone:
            validation_note += " Phone validation is DISABLED - do not send phone OTP. "
        if validation_note:
            flow_instruction += validation_note
        
        prompt_text = f"""You are a {profession} assistant. Use ONLY the context below.

RULES:
1. Match user input to lead types, services, FAQs from context - use exact values
2. Use the greeting from context ONLY for the very first message - do NOT repeat it
3. {flow_instruction}
4. OPTION FORMAT: {option_format}
5. CONVERSATION FLOW TRACKING (CRITICAL):
   - Analyze conversation history CAREFULLY to determine what information has been collected
   - Extract collected info: leadType, serviceType, leadName, leadEmail, leadPhoneNumber, title (title is the text from the selected lead type)
   - Flow progression (STRICT ORDER - DO NOT SKIP STEPS):
     * Step 1: If NO lead type collected → ask for lead type and show ALL lead types using {option_format}
     * Step 2: If lead type collected but NO service type → MANDATORY: Ask "Which service are you interested in?" and show ALL services from context using {option_format} (ALL services must be shown)
       ** CRITICAL: Service type is REQUIRED for ALL lead types including "callback", "appointment arrangement", "further information" - DO NOT SKIP THIS STEP
       ** You MUST ask for service type immediately after lead type is selected, even for callback requests
       ** NEVER go directly from lead type to name - service type is ALWAYS required
     * Step 3: If service type collected but NO name → ask for name
     * Step 4: If name collected but NO email → ask for email
     * Step 5: If email collected but NO phone (and not WhatsApp) → ask for phone
     * Step 6: If all info collected AND all required OTP verifications complete (verify from conversation history) → generate JSON immediately
   - NEVER skip Step 2 (service type) - it is MANDATORY for every conversation
   - CRITICAL: When showing service options, you MUST show ALL services listed in the context, not just one or a few
   - For WhatsApp: Use numbered list format (1. Option 1, 2. Option 2, etc.)
   - For Web: Use button format (<button> Option Text </button> or <button value="value"> Text </button>)
   - Do NOT repeat questions already asked
   - Do NOT restart the conversation - continue from where you left off
   - If user provides information (name, email, phone), acknowledge and move to next step
5. OTP HANDLING (Use semantic understanding - detect user intent):
   - If user wants to resend OTP to the SAME contact (lost code, didn't receive, send again, resend, didn't get it, can't find code) → respond with ONLY: RETRY_OTP_REQUESTED
   - If user wants to CHANGE phone number (wrong number, different phone, send to another phone, new phone, that's not my number) → respond with ONLY: CHANGE_PHONE_REQUESTED
   - If user wants to CHANGE email address (wrong email, different email, send to another email, new email, that's not my email) → respond with ONLY: CHANGE_EMAIL_REQUESTED
   - Use semantic understanding to detect intent - analyze what the user means, not exact words
   - These special responses are REQUIRED - do NOT add any other text
6. When all info collected, output ONLY JSON: {json_fields}
7. Be conversational and helpful

{conversation_history}

Context (lead types, services, FAQs, greeting, validation flags):
{context}

User: {query}

Response:"""
        
        try:
            # Use async LLM call
            response = await self.llm.ainvoke(prompt_text)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error in direct LLM call: {e}")
            return ""
    
    async def get_initial_greeting(self, profession: str = "Clinic", is_whatsapp: bool = False, context_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get initial greeting using LangChain - uses greeting from context"""
        if not self.llm or not self.retriever:
            logger.warning("LLM or retriever not initialized")
            return None
        
        try:
            # Retrieve integration/greeting documents
            query = "greeting message initial"
            # Retrieve documents
            docs = self._retrieve_documents(query)
            
            # Format context
            context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
            
            # Get greeting from context_data - default if null/empty
            default_greeting = "Hi! How can I help you today?"
            greeting = default_greeting
            if context_data:
                integration = context_data.get("integration", {})
                context_greeting = integration.get("greeting", "")
                # Use greeting from context only if it's not null/empty
                if context_greeting and str(context_greeting).strip():
                    greeting = str(context_greeting).strip()
                else:
                    greeting = default_greeting
            
            # Get lead types from context
            lead_types = context_data.get("lead_types", []) if context_data else []
            lead_types_text = ""
            if lead_types:
                lead_types_text = "\nAvailable Lead Types:\n"
                for i, lt in enumerate(lead_types, 1):
                    if isinstance(lt, dict):
                        text = lt.get("text", "")
                        value = lt.get("value", "")
                        lead_types_text += f"{i}. {text} (value: {value})\n"
            
            prompt_text = f"""You are a {profession} assistant. Generate an initial greeting.

RULES:
1. Use this greeting: "{greeting}"
2. Present lead type options as buttons using EXACT format: <button>Option Text</button>
   - CRITICAL: NO spaces inside angle brackets - use <button> NOT < button >
   - CRITICAL: NO spaces before closing tag - use </button> NOT < /button >
   - Example CORRECT: <button>I would like a call back</button>
   - Example WRONG: < button >I would like a call back< /button >
3. Be warm and welcoming
4. Include all lead types from the list below
5. Format must be EXACT: <button>Text</button> with NO spaces in tags

Lead Types:
{lead_types_text}

Context:
{context}

Generate the initial greeting with lead type buttons. Use EXACT format: <button>Option Text</button> with NO spaces inside the angle brackets:"""
            
            response = await self.llm.ainvoke(prompt_text)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"Generated initial greeting using LangChain")
            return answer.strip() if answer else None
            
        except Exception as e:
            logger.error(f"Error getting initial greeting: {e}")
            return None
    
    def clear_vector_store(self) -> None:
        """Clear the current vector store"""
        self.vector_store = None
        self.retriever = None
        logger.info("Vector store cleared")

