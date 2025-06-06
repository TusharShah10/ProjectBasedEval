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
    return """You are an expert in evaluation criteria design for architectural and engineering proposal assessments. Your task is to refine evaluation prompts and create realistic mock extractions for testing.

CONTEXT: We evaluate design firm submissions for masterplan proposals including masterplan/landscape reports, engineering/infrastructure reports, cost schedules, and material schedules against ~300 metrics across categories and sub-categories.

TASK 1 - REFINE EVALUATION PROMPT:
Create ULTRA-PRECISE scoring criteria that eliminate all vagueness. Each criterion must specify:

CRITICAL REQUIREMENTS FOR REFINED CRITERIA:
1. **MATCH THE SCORING FORMAT** from the provided refined prompt examples exactly
   - If examples use ranges (e.g., "0, 10-49%, 60-99%, 100%") then use the same range format
   - If examples use absolute percentages (e.g., "0%, 25%, 50%, 75%, 100%") then use the same absolute format
   - If examples use other scoring systems, replicate that exact system
   - Maintain consistency with the example format throughout all criteria

2. Define EXACTLY what constitutes different levels of detail for each score band
3. Specify PRECISE thresholds (e.g., "must include at least 3 specific examples" or "must provide quantitative measurements")
4. Use MEASURABLE language (e.g., "detailed specifications including materials, dimensions, and technical standards" not just "adequate specifications")
5. Create CLEAR differentiation between score bands - no overlap or ambiguity
6. Include SPECIFIC examples of what qualifies for each score level
7. Define BOTH content requirements AND quality/detail requirements for each criterion

EXAMPLE OF PRECISE CRITERION (format will vary based on training examples):
Instead of: "Sustainability measures are addressed"
Use format that matches the training examples provided, such as:

Format A (if examples use ranges):
"Sustainability measures: 
- 85-100%: Includes specific sustainability certifications (LEED, BREEAM), quantified energy reduction targets (%), detailed renewable energy systems with capacity specifications
- 50-84%: Includes general sustainability goals, mentions 2-3 specific green technologies with basic specifications
- 25-49%: Mentions sustainability concepts but lacks specific technologies or quantified targets
- 0-24%: No sustainability measures mentioned or only vague references"

Format B (if examples use absolute percentages):
"Sustainability measures:
- 100%: Comprehensive sustainability framework with specific certifications, quantified targets, and detailed implementation plans
- 75%: Strong sustainability approach with most certifications and clear targets
- 50%: Adequate sustainability measures with some specific elements
- 25%: Basic sustainability considerations with minimal detail
- 0%: No meaningful sustainability measures"

TASK 2 - CREATE MOCK PROJECT CONTEXT:
Generate concise extracted data from project documents that represents key project requirements with specific, measurable details.

TASK 3 - CREATE 3 MOCK SCENARIOS:
Generate scenarios that PRECISELY align with the scoring criteria format used:
- High score scenario: Must hit ALL criteria at the highest level with comprehensive detail
- Mid score scenario: Must hit criteria at the middle level with adequate but not comprehensive detail
- Low score scenario: Must clearly fall into the lowest range with missing or minimal detail

CRITICAL REQUIREMENTS:
- All outputs should be CONCISE but SPECIFIC
- Present as direct extractions, not commentary
- NO phrases like "The project lays out", "The proposal states", etc.
- Ensure scenarios PRECISELY match the scoring criteria definitions
- Make quality and detail level differences CRYSTAL CLEAR between scenarios
- Use the EXACT scoring format from the training examples

OUTPUT FORMAT:
## REFINED EVALUATION PROMPT
[Ultra-precise scoring criteria using the EXACT format from training examples]

## MOCK PROJECT CONTEXT
**Extracted Project Requirements:**
[Specific, measurable project requirements]

## MOCK SCENARIOS

### [Highest Score]% Score Scenario
**Extracted Submission Data:**
[Data with comprehensive detail matching highest score criteria exactly]

### [Middle Score]% Score Scenario  
**Extracted Submission Data:**
[Data with adequate detail matching middle score criteria exactly]

### [Lowest Score]% Score Scenario
**Extracted Submission Data:**
[Data with minimal/missing detail matching lowest score criteria exactly]
"""

# Function to get refined evaluation prompt and scenarios with project context
def get_evaluation_refinement(evaluation_prompt, submission_extraction_prompt, project_extraction_prompt, example_extraction_outputs, refined_prompt_examples, selected_model, gemini_client, openai_client):
    system_prompt = create_system_prompt()
    
    full_prompt = f"""{system_prompt}

ORIGINAL EVALUATION PROMPT TO REFINE:
{evaluation_prompt}

SUBMISSION EXTRACTION PROMPT (How data is extracted from submissions):
{submission_extraction_prompt}

PROJECT EXTRACTION PROMPT (How data is extracted from project documents):
{project_extraction_prompt}

EXAMPLE EXTRACTION OUTPUTS (Format reference):
{example_extraction_outputs if example_extraction_outputs else "No examples provided"}

EXAMPLES OF REFINED EVALUATION PROMPTS (CRITICAL - Use these to determine the EXACT scoring format to replicate):
{refined_prompt_examples if refined_prompt_examples else "No refined prompt examples provided - use default 85-100%, 50-84%, 25-49%, 0-24% format"}

IMPORTANT: Analyze the refined prompt examples carefully to identify their scoring format (ranges vs absolute percentages vs other systems) and replicate that EXACT format in your refined evaluation criteria. The scoring format must be consistent with the examples provided.

Create ultra-precise evaluation criteria using the EXACT scoring format from the examples, mock project context, and three mock submission scenarios that match the scoring criteria definitions. Eliminate ALL vagueness - every criterion must have clear, measurable requirements for each score band."""
    
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

# Function to validate scenarios using Gemini 
def validate_scenarios_with_gemini(refined_criteria, project_context, scenarios, gemini_client):
    """Validate that scenarios score as expected using Gemini"""
    
    validation_results = {}
    
    for target_score, scenario_data in scenarios.items():
        validation_prompt = f"""
Using the ULTRA-PRECISE evaluation criteria below, score the submission data against the project requirements. 

IMPORTANT: Follow the criteria EXACTLY as written. Pay close attention to the specific thresholds and detail requirements for each score band.

EVALUATION CRITERIA:
{refined_criteria}

PROJECT REQUIREMENTS:
{project_context}

SUBMISSION DATA TO EVALUATE:
{scenario_data}

Carefully compare the submission data against the project requirements using the precise scoring criteria. For each criterion, determine which score band the submission data falls into based on the SPECIFIC requirements defined.

Respond with only the numerical score (0-100) that reflects the overall evaluation.
"""
        
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=[validation_prompt]
            )
            
            # Extract score from response
            score_text = response.text.strip()
            score = None
            
            # Try to extract numerical score
            import re
            score_match = re.search(r'(\d+)', score_text)
            if score_match:
                score = int(score_match.group(1))
                if 0 <= score <= 100:
                    # Adjust tolerance based on target score
                    if target_score == 0:
                        tolerance = 25  # Low score range
                    elif target_score == 50:
                        tolerance = 20  # Mid score range
                    else:  # target_score == 100
                        tolerance = 15  # High score range
                    
                    validation_results[target_score] = {
                        'expected': target_score,
                        'actual': score,
                        'status': 'success' if abs(score - target_score) <= tolerance else 'needs_adjustment'
                    }
                else:
                    validation_results[target_score] = {'expected': target_score, 'actual': None, 'status': 'error'}
            else:
                validation_results[target_score] = {'expected': target_score, 'actual': None, 'status': 'error'}
                
        except Exception as e:
            validation_results[target_score] = {'expected': target_score, 'actual': None, 'status': 'error', 'error': str(e)}
    
    return validation_results

# Function to extract scenarios from result text
def extract_scenarios_from_result(result_text):
    """Extract individual scenarios from the generated result"""
    scenarios = {}
    
    # Look for scenario sections with flexible score matching
    lines = result_text.split('\n')
    current_scenario = None
    current_data = []
    
    for line in lines:
        # Look for patterns like "100% Score Scenario", "75% Score Scenario", etc.
        score_match = re.search(r'(\d+)%?\s*Score\s*Scenario', line, re.IGNORECASE)
        if score_match:
            # Save previous scenario if exists
            if current_scenario is not None and current_data:
                scenarios[current_scenario] = '\n'.join(current_data).strip()
            
            # Start new scenario
            current_scenario = int(score_match.group(1))
            current_data = []
        elif current_scenario is not None and line.strip() and not line.startswith('**') and not line.startswith('###'):
            current_data.append(line)
    
    # Don't forget the last scenario
    if current_scenario is not None and current_data:
        scenarios[current_scenario] = '\n'.join(current_data).strip()
    
    return scenarios

# Function to extract project context from result
def extract_project_context_from_result(result_text):
    """Extract project context from the generated result"""
    lines = result_text.split('\n')
    in_project_section = False
    project_data = []
    
    for line in lines:
        if 'MOCK PROJECT CONTEXT' in line:
            in_project_section = True
            continue
        elif 'MOCK SCENARIOS' in line:
            in_project_section = False
            break
        elif in_project_section and line.strip() and not line.startswith('**'):
            project_data.append(line)
    
    return '\n'.join(project_data).strip()

# Function to extract refined criteria from result
def extract_refined_criteria_from_result(result_text):
    """Extract refined criteria from the generated result"""
    lines = result_text.split('\n')
    in_criteria_section = False
    criteria_data = []
    
    for line in lines:
        if 'REFINED EVALUATION PROMPT' in line:
            in_criteria_section = True
            continue
        elif 'MOCK PROJECT CONTEXT' in line:
            in_criteria_section = False
            break
        elif in_criteria_section and line.strip() and not line.startswith('##'):
            criteria_data.append(line)
    
    return '\n'.join(criteria_data).strip()

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
    
    # Add validation toggle
    st.sidebar.header("Validation Options")
    enable_validation = st.sidebar.checkbox("Enable scenario validation with Gemini 2.0 flash lite", value=True)
    
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
        placeholder="Enter the prompt that describes how data is extracted from submission documents..."
    )

    # Input for project extraction prompt
    st.header("🏗️ Input Project Extraction Prompt")
    project_extraction_prompt = st.text_area(
        "Enter the project extraction prompt:",
        height=150,
        placeholder="Enter the prompt that describes how to extract key requirements from project documents..."
    )

    # Input for example extraction outputs
    st.header("🗒️ Input Example Extraction Outputs (Optional)")
    example_extraction_outputs = st.text_area(
        "Enter example extraction outputs:",
        height=150,
        placeholder="Enter examples of extracted data format..."
    )
    
    # Training examples section - UPDATED
    st.header("📚 Refined Evaluation Prompt Examples (Optional)")
    st.write("Provide examples of well-refined evaluation prompts. **The scoring format from these examples will be used for the output:**")
    refined_prompt_examples = st.text_area(
        "Refined Prompt Examples:",
        height=150,
        placeholder="Enter examples of refined evaluation prompts. The tool will match the scoring format used in these examples (e.g., if you use ranges like '0, 10-49%, 60-99%, 100%' or absolute percentages like '0%, 25%, 50%, 75%, 100%')..."
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
                    example_extraction_outputs, refined_prompt_examples, selected_model, gemini_client, openai_client
                )
                
                # Validate scenarios if enabled
                validation_results = None
                if enable_validation:
                    with st.spinner("Validating scenarios with Gemini..."):
                        # Extract components from result
                        refined_criteria = extract_refined_criteria_from_result(result)
                        project_context = extract_project_context_from_result(result)
                        scenarios = extract_scenarios_from_result(result)
                        
                        if refined_criteria and project_context and scenarios:
                            validation_results = validate_scenarios_with_gemini(
                                refined_criteria, project_context, scenarios, gemini_client
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
                
                # Validation results
                if validation_results:
                    st.header("🎯 Validation Results")
                    
                    # Create columns dynamically based on number of scenarios
                    num_scenarios = len(validation_results)
                    cols = st.columns(num_scenarios)
                    
                    for i, (target_score, result_data) in enumerate(validation_results.items()):
                        with cols[i]:
                            if result_data['status'] == 'success':
                                st.metric(
                                    f"{target_score}% Scenario", 
                                    f"{result_data['actual']}%",
                                    delta=f"{result_data['actual'] - target_score}%"
                                )
                                st.success("✅ Within range")
                            elif result_data['status'] == 'needs_adjustment':
                                st.metric(
                                    f"{target_score}% Scenario", 
                                    f"{result_data['actual']}%",
                                    delta=f"{result_data['actual'] - target_score}%"
                                )
                                st.warning("⚠️ Score deviation")
                            else:
                                st.metric(f"{target_score}% Scenario", "Error")
                                st.error("❌ Validation failed")
                
                # Main output
                st.header("📋 Results")
                st.markdown(result)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Instructions section
    with st.expander("📖 How to Use This Tool"):
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Enter Evaluation Prompt**: Your current evaluation criteria
        2. **Enter Submission Extraction Prompt**: How data is extracted from submissions
        3. **Enter Project Extraction Prompt**: How key requirements are extracted from project documents
        4. **Add Example Extraction Outputs** (Optional): Sample extraction formats
        5. **Add Refined Prompt Examples** (Optional): Examples that define the scoring format to use
        6. **Enable Validation**: Automatically test scenarios with Gemini 2.0 Flash Lite
        7. **Select Model**: Choose between Gemini or OpenAI for generation
        8. **Click Process**: Generate ultra-precise criteria and validated scenarios
        
        ### Scoring Format Matching:
        The tool will automatically detect and replicate the scoring format from your refined prompt examples:
        - **Range Format**: "0, 10-49%, 60-99%, 100%" → Uses same ranges
        - **Absolute Format**: "0%, 25%, 50%, 75%, 100%" → Uses same percentages  
        - **Custom Format**: Any other system → Replicates your exact format
        - **No Examples**: Defaults to "85-100%, 50-84%, 25-49%, 0-24%" format
        
        ### Refined Prompt Examples Should Include:
        - **Your preferred scoring format** (this will be replicated exactly)
        - **Ultra-precise criteria** with specific requirements
        - **Measurable thresholds** for each score level
        - **Clear quality definitions** for different detail levels
        - **No vague language** - every term should be specific and actionable
        
        ### Output Includes:
        - **Ultra-Precise Evaluation Criteria**: Using your exact scoring format
        - **Mock Project Context**: Extracted key project requirements
        - **Mock Scenarios**: Scenarios matching your scoring format precisely
        - **Validation Results**: Actual scores achieved by scenarios when tested
        """)
    
if __name__ == "__main__":
    main()
