import streamlit as st
import openai
from google import genai
from google.genai import types
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients for each API
def initialize_clients():
    # Try to get from Streamlit secrets first (for cloud deployment)
    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except:
        # Fallback to environment variables (for local development)
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not gemini_api_key or not openai_api_key:
        st.error("API keys not found. Please check your environment variables or Streamlit secrets.")
        st.stop()
    
    # Gemini
    gemini_client = genai.Client(
        api_key=gemini_api_key,
        http_options=types.HttpOptions(base_url="https://google-gemini.prod.ai-gateway.quantumblack.com/d9d53c5f-5d10-4661-9ee8-4e0d8afb99e3")
    )
    
    # OpenAI
    openai_client = openai.OpenAI(
        api_key=openai_api_key, 
        base_url="https://openai.prod.ai-gateway.quantumblack.com/d9d53c5f-5d10-4661-9ee8-4e0d8afb99e3/v1"
    )
    
    return gemini_client, openai_client

# Function to calculate cost
def calculate_cost(input_tokens, output_tokens, input_rate, output_rate):
    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    return input_cost + output_cost

# Function to create the system prompt for evaluation refinement
def create_system_prompt():
    return """You are an expert in evaluation criteria design for architectural and engineering proposal assessments. Your task is to refine evaluation prompts and create realistic mock scenarios for testing.

CONTEXT: We evaluate design firm submissions for masterplan proposals including masterplan/landscape reports, engineering/infrastructure reports, cost schedules, and material schedules against ~300 metrics across categories and sub-categories.

TASK 1 - REFINE EVALUATION PROMPT:
1. Restructure the scoring system with clear breakdowns:
   - 0-49% (0, 1-20, 21-49): Poor/Missing
   - 50-85% (50-60, 61-85): Adequate/Good  
   - 86-100% (86-99, 100): Excellent/Perfect

2. Make criteria specific, measurable, and clear
3. Ensure the prompt guides LLM scoring effectively
4. Maintain professional evaluation language

TASK 2 - CREATE MOCK PROJECT CONTEXT:
Generate a realistic project context that would attract design firm submissions. This should include project requirements, scope, and specifications that the submissions would be responding to.

TASK 3 - CREATE 3 MOCK SCENARIOS:
Generate realistic extraction data examples that would score:
- 100%: Perfect, comprehensive data meeting all criteria
- 50%: Adequate data with some gaps or limitations
- 0%: Missing, irrelevant, or completely inadequate data

REQUIREMENTS FOR MOCK PROJECT CONTEXT:
- Create a realistic architectural/engineering project that firms would submit proposals for
- Include specific project requirements, scope, location details, and constraints
- Make it detailed enough that submission scenarios can be contextually relevant
- Present as a project brief or RFP summary

REQUIREMENTS FOR MOCK SCENARIOS:
- Base the mock scenarios on both the submission extraction prompt and the mock project context
- Present as direct extractions from submission documents responding to the specific project
- Use factual, technical language 
- Make data detailed and realistic for architectural/engineering proposals
- Ensure clear differentiation between score levels
- Format as if extracted from actual proposal documents using the extraction methodology described
- Always keep this as normal text, do not add any complicated formatting to this at all
- For any of the mock scenarios- (avoid evaluative phrases like "lacks", "does not have" etc) AVOID AT ANY COST
- For the 100% scenario, give mock examples which would match the criteria defined above 100% and perfectly
- For any of the scenarios, especially the 0 percent one- there will be mentions of the items from the Submission Extraction Prompt, these would not all be "no data" or the equivalent 
- THIS IS AN EXTRACTION and not a commentary. DO NOT add things like "The plan lays out" or "The master plan says" or "the proposal lays out" or other such statements. Be smart, this is an extraction not a summary. It should seem like this is directly there in the proposal, when you extract it, it would not say "The proposal"
- Good scenarios include things like "Materials will be selected in accordance with budget constraints."
- STRICTLY Do not use double quotes in scenarios you are generating, it should be just plain text

OUTPUT FORMAT:
## REFINED EVALUATION PROMPT
[Refined prompt with clear scoring structure]

## MOCK PROJECT CONTEXT
[Detailed project context that submissions would be responding to]

## MOCK SCENARIOS

### 100% Score Scenario
**Extracted Data:**
[Detailed extraction that perfectly meets criteria, based on the extraction prompt methodology and project context]

### 50% Score Scenario  
**Extracted Data:**
[Partial extraction with some gaps, based on the extraction prompt methodology and project context]

### 0% Score Scenario
**Extracted Data:**
[Missing/inadequate extraction, based on the extraction prompt methodology and project context]
"""

# Function to get refined evaluation prompt and scenarios with project context
def get_evaluation_refinement(evaluation_prompt, submission_extraction_prompt, project_extraction_prompt, example_extraction_outputs, training_examples, selected_model, gemini_client, openai_client):
    system_prompt = create_system_prompt()
    
    full_prompt = f"""{system_prompt}

ORIGINAL EVALUATION PROMPT TO REFINE:
{evaluation_prompt}

SUBMISSION EXTRACTION PROMPT (Use this to understand how data is extracted and create realistic mock scenarios):
{submission_extraction_prompt}

PROJECT EXTRACTION PROMPT (Use this to understand how to create the mock project context):
{project_extraction_prompt}

EXAMPLE EXTRACTION OUTPUTS (Use these to understand the desired format for mock scenarios):
{example_extraction_outputs if example_extraction_outputs else "No examples provided"}

TRAINING EXAMPLES (Examples of refined prompts):
{training_examples if training_examples else "No training examples provided"}

Please provide the refined evaluation prompt, mock project context, and three mock scenarios as specified above. The mock scenarios should reflect realistic extracted data that would result from applying the submission extraction prompt to actual design firm documents responding to the specific project context."""
    
    start_time = time.time()
    
    if selected_model == "Gemini":
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[full_prompt]
        )
        result = response.text
    elif selected_model == "OpenAI":
        response = openai_client.chat.completions.create(
            messages=[{"role": "user", "content": full_prompt}],
            model="gpt-4o-mini"
        )
        result = response.choices[0].message.content
    
    end_time = time.time()
    
    # Calculate approximate cost
    input_tokens = len(full_prompt.split())
    output_tokens = len(result.split())
    
    if selected_model == "Gemini":
        cost = calculate_cost(input_tokens, output_tokens, 0.075, 0.3)
    else:  # OpenAI
        cost = calculate_cost(input_tokens, output_tokens, 0.15, 0.6)
    
    return result, end_time - start_time, cost

# Streamlit UI
def main():
    st.title("🔍 Evaluation Prompt Refinement Tool")
    st.write("Refine evaluation prompts and generate mock scenarios for design firm submission assessments")
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose AI Model:",
        ["Gemini", "OpenAI"]
    )
    
    # Main input section
    st.header("📝 Input Evaluation Prompt")
    evaluation_prompt = st.text_area(
        "Enter your evaluation prompt with criteria:",
        height=200,
        placeholder="Enter the evaluation prompt that contains the evaluation criteria you want to refine..."
    )
    
    # Input for submission extraction prompt
    st.header("📄 Input Submission Extraction Prompt")
    submission_extraction_prompt = st.text_area(
        "Enter the submission extraction prompt:",
        height=150,
        placeholder="Enter the prompt that describes how data is extracted from submission documents. This will be used to create realistic mock scenarios..."
    )

    # NEW: Input for project extraction prompt
    st.header("🏗️ Input Project Extraction Prompt")
    project_extraction_prompt = st.text_area(
        "Enter the project extraction prompt:",
        height=150,
        placeholder="Enter the prompt that describes how to create mock project contexts that would attract design firm submissions..."
    )

    # Input for example extraction outputs
    st.header("🗒️ Input Example Extraction Outputs (Optional)")
    example_extraction_outputs = st.text_area(
        "Enter example extraction outputs:",
        height=150,
        placeholder="Enter one or more example extraction outputs that show the desired format for the mock scenarios..."
    )
    
    # Training examples section - moved to main area
    st.header("📚 Training Examples (Optional)")
    st.write("Provide examples of how you've manually refined prompts before:")
    training_examples = st.text_area(
        "Training Examples:",
        height=150,
        placeholder="Example: Old prompt -> New refined prompt..."
    )
    
    # Additional context
    with st.expander("🏗️ Additional Context"):
        document_types = st.multiselect(
            "Select relevant document types:",
            ["Masterplan Report", "Landscape Report", "Engineering Report", 
             "Infrastructure Report", "Cost Schedule", "Material Schedule"],
            default=["Masterplan Report"]
        )
        
        metric_category = st.text_input(
            "Metric Category (optional):",
            placeholder="e.g., Sustainability, Design Quality, Technical Compliance"
        )
    
    # Process button
    if st.button("🚀 Refine Prompt & Generate Scenarios", type="primary"):
        if not evaluation_prompt.strip():
            st.error("Please enter an evaluation prompt to refine.")
            return
        if not submission_extraction_prompt.strip():
            st.error("Please enter a submission extraction prompt.")
            return
        if not project_extraction_prompt.strip():
            st.error("Please enter a project extraction prompt.")
            return
        
        with st.spinner(f"Processing with {selected_model}..."):
            try:
                # Initialize clients
                gemini_client, openai_client = initialize_clients()
                
                # Get refinement
                result, processing_time, cost = get_evaluation_refinement(
                    evaluation_prompt, submission_extraction_prompt, project_extraction_prompt, 
                    example_extraction_outputs, training_examples, selected_model, gemini_client, openai_client
                )
                
                # Display results
                st.success("✅ Refinement completed!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Used", selected_model)
                with col2:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                with col3:
                    st.metric("Estimated Cost", f"${cost:.4f}")
                
                # Main output
                st.header("📋 Results")
                st.markdown(result)
                
                # Download option
                st.download_button(
                    label="📥 Download Results",
                    data=result,
                    file_name="refined_evaluation_prompt.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Instructions section
    with st.expander("📖 How to Use This Tool"):
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Enter Evaluation Prompt**: Paste your current evaluation prompt that contains the criteria you want to refine
        
        2. **Enter Submission Extraction Prompt**: Provide the prompt that describes how data is extracted from submission documents. This helps create realistic mock scenarios.
        
        3. **Enter Project Extraction Prompt**: Provide the prompt that describes how to create mock project contexts that would attract design firm submissions.
        
        4. **Add Example Extraction Outputs** (Optional): Paste in one or more examples of actual extraction outputs so the tool learns the correct format.
        
        5. **Add Training Examples** (Optional): Provide examples of how you've manually refined prompts before to guide the refinement process.
        
        6. **Select Model**: Choose between Gemini or OpenAI for processing
        
        7. **Add Context** (Optional): 
           - Specify relevant document types
           - Add metric category information
        
        8. **Click Process**: The tool will:
           - Refine your evaluation prompt with proper scoring structure (0-49%, 50-85%, 86-100%)
           - Generate a realistic mock project context that would attract submissions
           - Generate 3 realistic mock scenarios for testing (100%, 50%, 0% scores) based on your extraction methodology, the example outputs you provided, and the mock project context
        
        ### Output Includes:
        - **Refined Evaluation Prompt**: Restructured with clear scoring guidelines
        - **Mock Project Context**: Realistic project details that submissions would be responding to
        - **Mock Scenarios**: Realistic extraction data examples for each score level, created based on your extraction prompt, example formats, and project context
        - **Performance Metrics**: Processing time and estimated cost
        
        ### Key Improvements:
        - **Contextual Scenarios**: Mock scenarios are now generated in response to a specific project context, making them more realistic and relevant
        - **Project-Driven Approach**: The mock project context ensures that submission scenarios reflect realistic responses to actual project requirements
        - **Enhanced Realism**: By grounding scenarios in a specific project, the extracted data becomes more believable and useful for testing
        
        ### Tips for Best Results:
        - Provide clear, specific evaluation criteria in your input
        - Include detailed extraction methodology in the submission extraction prompt
        - Make the project extraction prompt specific about the types of projects you evaluate
        - Paste example extraction outputs so the generated scenarios match your desired format
        - Include training examples to guide the refinement process
        - Review the refined prompt and project context, and adjust as needed for your specific use case
        """)
    
if __name__ == "__main__":
    main()
