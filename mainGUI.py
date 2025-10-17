import streamlit as st
from PIL import Image
import os
import torch
import urllib.parse
from transformers import AutoTokenizer, AutoModelForCausalLM
from predictions import predict, analyze_image

# --- Set Streamlit Page Config FIRST ---
st.set_page_config(
    page_title="Bone Fracture Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern Medical Theme ---
st.markdown("""
<style>
    /* Medical Theme Colors */
    :root {
        --primary-blue: #1f77b4;
        --light-blue: #e3f2fd;
        --accent-blue: #2196f3;
        --success-green: #4caf50;
        --danger-red: #f44336;
        --text-dark: #2c3e50;
        --text-light: #7f8c8d;
        --bg-white: #ffffff;
        --bg-light: #f8f9fa;
    }

    /* Main Container Styling */
    .main-container {
        background: var(--bg-white);
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Card Styling */
    .card {
        background: var(--bg-white);
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }

    /* Image Card */
    .image-card {
        background: var(--bg-white);
        border-radius: 16px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        border: 2px solid var(--light-blue);
    }

    /* Badge Styling */
    .badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        text-align: center;
        display: inline-block;
        margin: 4px;
    }

    .badge-fracture {
        background: var(--danger-red);
        color: white;
    }

    .badge-normal {
        background: var(--success-green);
        color: white;
    }

    /* Chat Bubble Styling */
    .chat-bubble {
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        max-width: 80%;
    }

    .chat-bubble-user {
        background: var(--accent-blue);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }

    .chat-bubble-assistant {
        background: #f1f3f4;
        color: var(--text-dark);
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }

    /* Title Styling */
    .main-title {
        color: var(--primary-blue);
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 8px;
    }

    .subtitle {
        color: var(--text-light);
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 32px;
    }

    /* Button Styling */
    .stButton > button {
        background: var(--accent-blue);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: var(--primary-blue);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.3);
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background: var(--bg-light);
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background: var(--light-blue);
        border-radius: 8px;
        font-weight: 600;
        color: var(--primary-blue);
    }

    /* Map Container */
    .map-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemma model (CPU version)
@st.cache_resource
def load_gemma_model():
    os.environ["HF_TOKEN"] = "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"
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    return tokenizer, model

tokenizer, model = load_gemma_model()

# --- Initialize Session State ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_bone_type' not in st.session_state:
    st.session_state.last_bone_type = None
if 'image_processed' not in st.session_state:
    st.session_state.image_processed = False

# --- Header Section ---
st.markdown('<h1 class="main-title">🦴 Bone Fracture Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered X-ray analysis with medical assistant</p>', unsafe_allow_html=True)

# --- Sidebar for Controls ---
with st.sidebar:
    st.markdown("### 📋 Analysis Controls")
    st.markdown("---")
    
    user_city = st.text_input(
        "🏥 Enter your city",
        placeholder="e.g., New York, London",
        help="Enter your city for nearby hospital recommendations"
    )
    
    analyze_button = st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True,
        help="Click to analyze the uploaded X-ray image"
    )

# --- Main Content Area ---
# --- Image Upload and Display Section ---
st.markdown("### 📸 X-ray Image Upload")

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["png", "jpg", "jpeg"],
    help="Upload a clear X-ray image of elbow, hand, or shoulder"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
        
    # Display image in a styled card
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
        
    temp_path = "temp_uploaded_image.png"
    image.save(temp_path)

    # Analysis button moved to sidebar, but we need to trigger it here
    if analyze_button:
        with st.spinner("🔬 Analyzing X-ray image..."):
            try:
                bone_type_result = predict(temp_path)
                if bone_type_result not in ["Elbow", "Hand", "Shoulder"]:
                    st.error("❌ Image doesn't match expected bone types (Elbow, Hand, Shoulder).")
                    st.session_state.image_processed = False
                else:
                    result = predict(temp_path, bone_type_result)
                    st.session_state.last_prediction = result
                    st.session_state.last_bone_type = bone_type_result
                    st.session_state.image_processed = True
                    
                    # Display result with colored badge
                    if result == 'fractured':
                        st.markdown(f'<div class="badge badge-fracture">🚨 FRACTURED {bone_type_result.upper()}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="badge badge-normal">✅ NORMAL {bone_type_result.upper()}</div>', unsafe_allow_html=True)
                    
                    # Add initial bot message with analysis
                    initial_msg = f"I've analyzed your {bone_type_result} X-ray. "
                    if result == 'fractured':
                        initial_msg += "It appears to be fractured. Please ask me any questions about treatment and care."
                    else:
                        initial_msg += "No fracture was detected. Feel free to ask me any questions."
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": initial_msg})

                    st.success("✅ Analysis complete! Check the results below.")
                        
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.session_state.image_processed = False
else:
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    st.image("images/Question_Mark.jpg", caption="Awaiting X-ray upload", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.session_state.image_processed = False

# --- Analysis Results Section (Below X-ray) ---
if st.session_state.image_processed and st.session_state.last_prediction:
    st.markdown("---")
    st.markdown("### 📊 Analysis Results")
    
    # Structured analysis
    structured = analyze_image(temp_path)
    
    # Results in equal columns
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🔍 Detailed Findings", expanded=True):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Fracture status
            if structured.get("fracture_present"):
                st.markdown('<div class="badge badge-fracture">🚨 FRACTURE DETECTED</div>', unsafe_allow_html=True)
                hospital_department = "🏥 Orthopedic Hospital / Trauma Care Center"
            else:
                st.markdown('<div class="badge badge-normal">✅ NO FRACTURE</div>', unsafe_allow_html=True)
                hospital_department = "🏥 Orthopedic OPD (if symptoms persist)"
            
            st.markdown("---")
            
            # Structured data
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Bone Type", st.session_state.last_bone_type)
                st.metric("Fracture Type", structured.get("fracture_type", "N/A"))
            
            with col_b:
                severity = structured.get("severity_percent")
                if severity is not None:
                    st.metric("Severity", f"{severity}%")
                else:
                    st.metric("Severity", "N/A")
                st.metric("Recommendation", hospital_department)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Hospital recommendation and map
        st.markdown("### 🏥 Hospital Recommendations")
        
        if user_city.strip():
            # Create map URL for the entered city
            query = f"Orthopedic hospital {user_city.strip()}"
            
            st.info(f"📍 Showing orthopedic hospitals near **{user_city.strip()}**")
            
            # Working Google Maps iframe - using a different approach
            maps_url = f"https://maps.google.com/maps?q={urllib.parse.quote_plus(query)}&t=&z=13&ie=UTF8&iwloc=&output=embed"
            
            # Display the map
            st.markdown("#### 📍 Interactive Map")
            st.components.v1.iframe(
                maps_url,
                height=300,
                scrolling=False
            )
            
        else:
            st.info("📍 Please enter a city name in the sidebar to see hospital recommendations and map.")
            
            # Default map showing general orthopedic hospitals
            st.markdown("#### 📍 General Orthopedic Hospitals")
            default_maps_url = "https://maps.google.com/maps?q=orthopedic+hospital&t=&z=13&ie=UTF8&iwloc=&output=embed"
            st.components.v1.iframe(
                default_maps_url,
                height=300,
                scrolling=False
            )
        

else:
    st.info("📋 Upload an X-ray and click 'Analyze Image' to see results here.")

# --- AI Assistant Chat Section ---
st.markdown("---")
st.markdown("### 🤖 Medical AI Assistant")

# Display chat history with styled bubbles
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-bubble chat-bubble-user">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble chat-bubble-assistant">{message["content"]}</div>', unsafe_allow_html=True)

# Show helpful tips if no chat history and image is processed
if len(st.session_state.chat_history) == 0 and st.session_state.image_processed:
    st.markdown("""
    <div class="chat-bubble chat-bubble-assistant">
    💡 <strong>Quick Tips:</strong> You can ask me about specific topics like:
    <br>• <strong>"all"</strong> - for complete comprehensive guide
    <br>• <strong>"suggestions"</strong> - for treatment and recovery tips
    <br>• <strong>"precautions"</strong> - for safety measures
    <br>• <strong>"food"</strong> - for dietary recommendations  
    <br>• <strong>"medicine"</strong> - for medication guidelines
    <br>• <strong>"exercise"</strong> - for physical activity recommendations
    <br>• <strong>"emergency"</strong> - for emergency signs and when to seek help
    <br>• <strong>"recovery"</strong> - for recovery timeline and progress
    </div>
    """, unsafe_allow_html=True)

# Chat input - This should be at the end
if prompt := st.chat_input("💬 Ask about your X-ray results..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Debug info (you can remove this later)
    st.write(f"Debug: Processing prompt: '{prompt}'")
    st.write(f"Debug: Image processed: {st.session_state.image_processed}")
    st.write(f"Debug: Last prediction: {st.session_state.last_prediction}")
    st.write(f"Debug: Last bone type: {st.session_state.last_bone_type}")
    
    # Generate response based on context
    if not st.session_state.image_processed:
        bot_response = """Please upload an X-ray image and click 'Analyze Image' first so I can help you.

💡 **What I can help you with once you upload an X-ray:**
• **"all"** - Complete comprehensive guide
• **"suggestions"** - Treatment and recovery tips
• **"precautions"** - Safety measures
• **"food"** - Dietary recommendations
• **"medicine"** - Medication guidelines
• **"exercise"** - Physical activity recommendations
• **"emergency"** - Emergency signs and when to seek help
• **"recovery"** - Recovery timeline and progress"""
    else:
        # Check for specific section keywords
        prompt_lower = prompt.lower()
        
        # Define section-specific responses
        section_responses = {
            "all": {
                "fractured": f"""
**📋 COMPREHENSIVE GUIDE FOR FRACTURED {st.session_state.last_bone_type.upper()}**

---

**💡 SUGGESTIONS & TREATMENT PLAN:**

• **Immediate Actions:**
  - Immobilize the affected area immediately
  - Apply ice pack for 15-20 minutes every 2-3 hours
  - Elevate the limb to reduce swelling
  - Seek emergency medical care

• **Medical Consultation:**
  - Visit orthopedic specialist within 24 hours
  - Get proper X-ray and imaging done
  - Follow doctor's treatment plan strictly

• **Recovery Tips:**
  - Keep the cast/splint dry and clean
  - Follow physical therapy recommendations
  - Avoid putting weight on the injured area
  - Attend all follow-up appointments

---

**⚠️ PRECAUTIONS & SAFETY MEASURES:**

• **Movement Restrictions:**
  - DO NOT move or put weight on the injured area
  - Avoid any twisting or bending motions
  - Keep the limb elevated when possible
  - Use crutches or sling as prescribed

• **Cast/Splint Care:**
  - Keep the cast completely dry
  - Don't insert objects inside the cast
  - Report any pain, swelling, or numbness
  - Don't cut or modify the cast yourself

• **Activity Limitations:**
  - Avoid driving until cleared by doctor
  - Don't participate in sports or heavy lifting
  - Follow specific activity restrictions
  - Attend all physical therapy sessions

---

**🍎 DIETARY RECOMMENDATIONS:**

• **Essential Nutrients:**
  - **Calcium**: Dairy products, leafy greens, fortified foods
  - **Vitamin D**: Fatty fish, egg yolks, sunlight exposure
  - **Protein**: Lean meats, fish, eggs, legumes
  - **Vitamin C**: Citrus fruits, berries, bell peppers

• **Foods to Include:**
  - Milk, yogurt, and cheese for calcium
  - Salmon, tuna for vitamin D and omega-3
  - Nuts and seeds for minerals
  - Dark leafy vegetables for vitamins

• **Foods to Avoid:**
  - Excessive caffeine (can interfere with calcium absorption)
  - High-sodium foods (can cause swelling)
  - Alcohol (slows healing process)
  - Processed foods with low nutritional value

• **Hydration:**
  - Drink 8-10 glasses of water daily
  - Include bone broth for collagen
  - Avoid sugary drinks

---

**💊 MEDICATION GUIDELINES:**

• **Pain Management:**
  - **Over-the-counter**: Acetaminophen (Tylenol) for pain
  - **Anti-inflammatory**: Ibuprofen (Advil) for swelling
  - **Prescription**: Follow doctor's recommendations for stronger pain relievers
  - **Topical**: Ice packs and elevation for natural relief

• **Important Notes:**
  - Take medications exactly as prescribed
  - Don't exceed recommended dosages
  - Inform doctor of all medications you're taking
  - Report any side effects immediately

• **Medications to Avoid:**
  - Don't take aspirin without doctor's approval
  - Avoid blood thinners unless prescribed
  - Be cautious with herbal supplements
  - Don't mix medications without consultation

• **Follow-up Care:**
  - Attend all medical appointments
  - Keep track of medication schedule
  - Report any unusual symptoms
  - Follow rehabilitation protocols

---

**🏥 EMERGENCY SIGNS TO WATCH FOR:**
- Severe pain that doesn't improve with medication
- Numbness or tingling in fingers/toes
- Blue or cold extremities
- Fever or signs of infection
- Difficulty breathing or chest pain

**📞 WHEN TO CALL EMERGENCY:**
- If you experience any emergency signs
- If the cast becomes too tight or uncomfortable
- If you notice any unusual symptoms
- If pain becomes unbearable
                """,
                "normal": f"""
**📋 COMPREHENSIVE GUIDE FOR NORMAL {st.session_state.last_bone_type.upper()}**

---

**💡 SUGGESTIONS & SELF-CARE:**

• **Self-Care Measures:**
  - Rest the affected area for 24-48 hours
  - Apply ice if there's any swelling
  - Gentle stretching exercises after 48 hours
  - Monitor for any changes in symptoms

• **Prevention Tips:**
  - Maintain good posture and ergonomics
  - Strengthen surrounding muscles
  - Use proper techniques during activities
  - Consider protective gear for sports

• **When to Seek Help:**
  - If pain persists beyond 48 hours
  - If swelling or bruising increases
  - If you experience numbness or tingling
  - If movement becomes more difficult

---

**⚠️ PRECAUTIONS & PREVENTION:**

• **Activity Modifications:**
  - Avoid repetitive strain movements
  - Take regular breaks during activities
  - Use proper ergonomics at work
  - Warm up before physical activities

• **Lifestyle Adjustments:**
  - Maintain good posture throughout the day
  - Avoid sudden jerky movements
  - Use supportive equipment when needed
  - Listen to your body's signals

• **Prevention Measures:**
  - Regular exercise to strengthen muscles
  - Proper nutrition for bone health
  - Adequate rest and recovery time
  - Regular check-ups with healthcare provider

---

**🍎 DIETARY RECOMMENDATIONS:**

• **Bone-Strengthening Foods:**
  - **Calcium-rich**: Dairy, fortified plant milk, leafy greens
  - **Vitamin D**: Fatty fish, mushrooms, fortified foods
  - **Magnesium**: Nuts, seeds, whole grains
  - **Vitamin K**: Green vegetables, fermented foods

• **Anti-Inflammatory Foods:**
  - Fatty fish (salmon, mackerel)
  - Berries and cherries
  - Turmeric and ginger
  - Olive oil and avocados

• **Foods to Limit:**
  - Processed foods high in salt
  - Sugary snacks and beverages
  - Excessive alcohol consumption
  - Foods high in saturated fats

• **Hydration:**
  - Maintain adequate water intake
  - Include herbal teas for antioxidants
  - Avoid excessive caffeine

---

**💊 MEDICATION & SUPPLEMENTS:**

• **Pain Relief (if needed):**
  - **Mild pain**: Acetaminophen (Tylenol)
  - **Inflammation**: Ibuprofen (Advil) or Naproxen
  - **Topical**: Pain relief creams or gels
  - **Natural**: Arnica, turmeric supplements

• **Preventive Supplements:**
  - **Calcium**: 1000-1200mg daily (consult doctor)
  - **Vitamin D**: 600-800 IU daily
  - **Omega-3**: Fish oil supplements
  - **Glucosamine**: For joint health (consult doctor)

• **Important Guidelines:**
  - Consult healthcare provider before starting supplements
  - Don't exceed recommended dosages
  - Monitor for any adverse reactions
  - Inform doctor of all medications and supplements

• **When to Seek Medical Help:**
  - If pain persists despite medication
  - If you experience side effects
  - If symptoms worsen
  - For prescription medication needs

---

**🏃‍♂️ EXERCISE & ACTIVITY RECOMMENDATIONS:**
- Gentle stretching exercises
- Low-impact activities like walking or swimming
- Strength training for surrounding muscles
- Flexibility exercises
- Proper warm-up and cool-down routines

**📊 MONITORING & FOLLOW-UP:**
- Regular self-assessment of symptoms
- Keep a pain and activity diary
- Schedule regular check-ups
- Monitor for any changes in condition
                """
            },
            "suggestions": {
                "fractured": f"""
**💡 Suggestions for Fractured {st.session_state.last_bone_type}:**

• **Immediate Actions:**
  - Immobilize the affected area immediately
  - Apply ice pack for 15-20 minutes every 2-3 hours
  - Elevate the limb to reduce swelling
  - Seek emergency medical care

• **Medical Consultation:**
  - Visit orthopedic specialist within 24 hours
  - Get proper X-ray and imaging done
  - Follow doctor's treatment plan strictly

• **Recovery Tips:**
  - Keep the cast/splint dry and clean
  - Follow physical therapy recommendations
  - Avoid putting weight on the injured area
  - Attend all follow-up appointments
                """,
                "normal": f"""
**💡 Suggestions for Normal {st.session_state.last_bone_type}:**

• **Self-Care Measures:**
  - Rest the affected area for 24-48 hours
  - Apply ice if there's any swelling
  - Gentle stretching exercises after 48 hours
  - Monitor for any changes in symptoms

• **Prevention Tips:**
  - Maintain good posture and ergonomics
  - Strengthen surrounding muscles
  - Use proper techniques during activities
  - Consider protective gear for sports

• **When to Seek Help:**
  - If pain persists beyond 48 hours
  - If swelling or bruising increases
  - If you experience numbness or tingling
  - If movement becomes more difficult
                """
            },
            "precautions": {
                "fractured": f"""
**⚠️ Precautions for Fractured {st.session_state.last_bone_type}:**

• **Movement Restrictions:**
  - DO NOT move or put weight on the injured area
  - Avoid any twisting or bending motions
  - Keep the limb elevated when possible
  - Use crutches or sling as prescribed

• **Cast/Splint Care:**
  - Keep the cast completely dry
  - Don't insert objects inside the cast
  - Report any pain, swelling, or numbness
  - Don't cut or modify the cast yourself

• **Activity Limitations:**
  - Avoid driving until cleared by doctor
  - Don't participate in sports or heavy lifting
  - Follow specific activity restrictions
  - Attend all physical therapy sessions
                """,
                "normal": f"""
**⚠️ Precautions for {st.session_state.last_bone_type} Health:**

• **Activity Modifications:**
  - Avoid repetitive strain movements
  - Take regular breaks during activities
  - Use proper ergonomics at work
  - Warm up before physical activities

• **Lifestyle Adjustments:**
  - Maintain good posture throughout the day
  - Avoid sudden jerky movements
  - Use supportive equipment when needed
  - Listen to your body's signals

• **Prevention Measures:**
  - Regular exercise to strengthen muscles
  - Proper nutrition for bone health
  - Adequate rest and recovery time
  - Regular check-ups with healthcare provider
                """
            },
            "food": {
                "fractured": f"""
**🍎 Dietary Recommendations for Fracture Recovery:**

• **Essential Nutrients:**
  - **Calcium**: Dairy products, leafy greens, fortified foods
  - **Vitamin D**: Fatty fish, egg yolks, sunlight exposure
  - **Protein**: Lean meats, fish, eggs, legumes
  - **Vitamin C**: Citrus fruits, berries, bell peppers

• **Foods to Include:**
  - Milk, yogurt, and cheese for calcium
  - Salmon, tuna for vitamin D and omega-3
  - Nuts and seeds for minerals
  - Dark leafy vegetables for vitamins

• **Foods to Avoid:**
  - Excessive caffeine (can interfere with calcium absorption)
  - High-sodium foods (can cause swelling)
  - Alcohol (slows healing process)
  - Processed foods with low nutritional value

• **Hydration:**
  - Drink 8-10 glasses of water daily
  - Include bone broth for collagen
  - Avoid sugary drinks
                """,
                "normal": f"""
**🍎 Dietary Recommendations for {st.session_state.last_bone_type} Health:**

• **Bone-Strengthening Foods:**
  - **Calcium-rich**: Dairy, fortified plant milk, leafy greens
  - **Vitamin D**: Fatty fish, mushrooms, fortified foods
  - **Magnesium**: Nuts, seeds, whole grains
  - **Vitamin K**: Green vegetables, fermented foods

• **Anti-Inflammatory Foods:**
  - Fatty fish (salmon, mackerel)
  - Berries and cherries
  - Turmeric and ginger
  - Olive oil and avocados

• **Foods to Limit:**
  - Processed foods high in salt
  - Sugary snacks and beverages
  - Excessive alcohol consumption
  - Foods high in saturated fats

• **Hydration:**
  - Maintain adequate water intake
  - Include herbal teas for antioxidants
  - Avoid excessive caffeine
                """
            },
            "medicine": {
                "fractured": f"""
**💊 Medication Guidelines for Fracture Recovery:**

• **Pain Management:**
  - **Over-the-counter**: Acetaminophen (Tylenol) for pain
  - **Anti-inflammatory**: Ibuprofen (Advil) for swelling
  - **Prescription**: Follow doctor's recommendations for stronger pain relievers
  - **Topical**: Ice packs and elevation for natural relief

• **Important Notes:**
  - Take medications exactly as prescribed
  - Don't exceed recommended dosages
  - Inform doctor of all medications you're taking
  - Report any side effects immediately

• **Medications to Avoid:**
  - Don't take aspirin without doctor's approval
  - Avoid blood thinners unless prescribed
  - Be cautious with herbal supplements
  - Don't mix medications without consultation

• **Follow-up Care:**
  - Attend all medical appointments
  - Keep track of medication schedule
  - Report any unusual symptoms
  - Follow rehabilitation protocols
                """,
                "normal": f"""
**💊 Medication Guidelines for {st.session_state.last_bone_type} Health:**

• **Pain Relief (if needed):**
  - **Mild pain**: Acetaminophen (Tylenol)
  - **Inflammation**: Ibuprofen (Advil) or Naproxen
  - **Topical**: Pain relief creams or gels
  - **Natural**: Arnica, turmeric supplements

• **Preventive Supplements:**
  - **Calcium**: 1000-1200mg daily (consult doctor)
  - **Vitamin D**: 600-800 IU daily
  - **Omega-3**: Fish oil supplements
  - **Glucosamine**: For joint health (consult doctor)

• **Important Guidelines:**
  - Consult healthcare provider before starting supplements
  - Don't exceed recommended dosages
  - Monitor for any adverse reactions
  - Inform doctor of all medications and supplements

• **When to Seek Medical Help:**
  - If pain persists despite medication
  - If you experience side effects
  - If symptoms worsen
  - For prescription medication needs
                """
            },
            "exercise": {
                "fractured": f"""
**🏃‍♂️ Exercise Guidelines for Fracture Recovery:**

• **During Cast/Splint Period:**
  - **Gentle Range of Motion**: Move unaffected joints
  - **Isometric Exercises**: Contract muscles without moving joints
  - **Breathing Exercises**: Deep breathing for circulation
  - **Core Strengthening**: Gentle abdominal exercises

• **Post-Cast Recovery:**
  - **Physical Therapy**: Follow prescribed exercise program
  - **Gradual Progression**: Start with gentle movements
  - **Strength Training**: Build muscle gradually
  - **Flexibility**: Gentle stretching as approved

• **Activities to Avoid:**
  - High-impact sports until cleared
  - Heavy lifting or weight training
  - Activities that cause pain
  - Contact sports

• **Safe Activities:**
  - Walking (if approved by doctor)
  - Swimming (after cast removal)
  - Stationary cycling
  - Gentle yoga (modified poses)
                """,
                "normal": f"""
**🏃‍♂️ Exercise Recommendations for {st.session_state.last_bone_type} Health:**

• **Strengthening Exercises:**
  - **Resistance Training**: Light weights or resistance bands
  - **Bodyweight Exercises**: Push-ups, squats, planks
  - **Functional Movements**: Daily activity simulations
  - **Core Strengthening**: Planks, bridges, bird dogs

• **Flexibility & Mobility:**
  - **Stretching**: Gentle daily stretching routine
  - **Yoga**: Modified poses for joint health
  - **Tai Chi**: Slow, controlled movements
  - **Pilates**: Core and flexibility focus

• **Cardiovascular Health:**
  - **Low-Impact Cardio**: Walking, swimming, cycling
  - **Interval Training**: Moderate intensity intervals
  - **Aerobic Activities**: Dancing, hiking
  - **Recovery**: Proper rest between sessions

• **Prevention Focus:**
  - **Balance Training**: Single-leg exercises
  - **Proprioception**: Balance board exercises
  - **Posture Work**: Ergonomic awareness
  - **Functional Fitness**: Real-world movement patterns
                """
            },
            "emergency": {
                "fractured": f"""
**🚨 Emergency Information for Fracture:**

• **Immediate Emergency Signs:**
  - Severe, unbearable pain
  - Numbness or tingling in extremities
  - Blue or cold fingers/toes
  - Difficulty breathing or chest pain
  - Fever above 101°F (38°C)
  - Signs of infection around cast

• **When to Call 911:**
  - Loss of consciousness
  - Severe bleeding
  - Difficulty breathing
  - Chest pain or pressure
  - Severe allergic reaction to medication

• **When to Go to ER:**
  - Cast becomes too tight or uncomfortable
  - Severe pain not relieved by medication
  - New numbness or weakness
  - Signs of infection (redness, warmth, pus)
  - Cast gets wet and can't be dried

• **Emergency Contacts:**
  - Primary care physician
  - Orthopedic specialist
  - Emergency room
  - Poison control (if medication issues)
                """,
                "normal": f"""
**🚨 Emergency Information for {st.session_state.last_bone_type} Health:**

• **Warning Signs to Watch For:**
  - Sudden, severe pain
  - Numbness or tingling
  - Swelling that doesn't improve
  - Redness or warmth in the area
  - Fever or chills
  - Difficulty moving the joint

• **When to Seek Immediate Care:**
  - Pain that interferes with daily activities
  - Symptoms that worsen over time
  - Signs of infection
  - Trauma or injury to the area
  - Persistent swelling or bruising

• **When to Call Your Doctor:**
  - Pain that persists beyond 48 hours
  - New symptoms develop
  - Difficulty with normal activities
  - Concerns about medication side effects
  - Questions about treatment plan

• **Prevention of Emergencies:**
  - Regular check-ups
  - Proper warm-up before activities
  - Use of protective equipment
  - Listening to body signals
  - Avoiding overexertion
                """
            },
            "recovery": {
                "fractured": f"""
**📈 Recovery Timeline for Fracture:**

• **Week 1-2 (Acute Phase):**
  - Focus on pain management
  - Keep cast/splint clean and dry
  - Elevate limb to reduce swelling
  - Follow doctor's medication schedule
  - Attend all medical appointments

• **Week 3-6 (Healing Phase):**
  - Continue cast care
  - Begin gentle range of motion (if approved)
  - Start physical therapy (when prescribed)
  - Monitor for any complications
  - Maintain good nutrition

• **Week 7-12 (Rehabilitation Phase):**
  - Cast removal (if applicable)
  - Intensive physical therapy
  - Gradual return to activities
  - Strength and flexibility training
  - Follow-up X-rays as scheduled

• **Month 3+ (Return to Activity):**
  - Full range of motion exercises
  - Sport-specific training
  - Gradual return to normal activities
  - Continued strength training
  - Regular follow-up appointments

• **Recovery Milestones:**
  - Pain-free movement
  - Full range of motion
  - Normal strength
  - Return to daily activities
  - Return to sports/work
                """,
                "normal": f"""
**📈 Recovery & Prevention Timeline:**

• **Immediate (0-48 hours):**
  - Rest the affected area
  - Apply ice if needed
  - Monitor symptoms
  - Avoid aggravating activities
  - Gentle stretching if approved

• **Short-term (1-2 weeks):**
  - Gradual return to activities
  - Continue gentle exercises
  - Monitor for any changes
  - Maintain good posture
  - Follow prevention guidelines

• **Long-term (1-3 months):**
  - Regular exercise routine
  - Strength training program
  - Flexibility maintenance
  - Lifestyle modifications
  - Regular check-ups

• **Ongoing Prevention:**
  - Maintain healthy lifestyle
  - Regular exercise
  - Proper nutrition
  - Stress management
  - Regular medical check-ups

• **Success Indicators:**
  - Pain-free movement
  - Normal range of motion
  - Good strength and flexibility
  - No recurring symptoms
  - Improved overall health
                """
            }
        }
        
        # Check if prompt contains section keywords
        detected_section = None
        
        # Keywords for each section
        section_keywords = {
            "all": ["all", "everything", "complete", "comprehensive", "full", "total", "guide", "overview", "summary"],
            "suggestions": ["suggestions", "suggestion", "tips", "advice", "recommendations", "what to do", "how to"],
            "precautions": ["precautions", "precaution", "safety", "avoid", "don't", "restrictions", "careful"],
            "food": ["food", "diet", "nutrition", "eating", "dietary", "meals", "supplements", "vitamins"],
            "medicine": ["medicine", "medication", "drugs", "pills", "pain", "relief", "treatment", "medicines"],
            "exercise": ["exercise", "workout", "training", "physical", "activity", "movement", "fitness", "sports"],
            "emergency": ["emergency", "urgent", "danger", "warning", "severe", "critical", "immediate", "help"],
            "recovery": ["recovery", "healing", "timeline", "progress", "rehab", "rehabilitation", "heal", "improve"]
        }
        
        # Check for section keywords
        for section, keywords in section_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected_section = section
                break
        
        if detected_section:
            # Provide section-specific response
            bot_response = section_responses[detected_section][st.session_state.last_prediction]
        else:
            # Create general context for Gemma model
            context = f"""
            You are a medical assistant helping a patient with their X-ray results.
            The patient's {st.session_state.last_bone_type} appears to be {st.session_state.last_prediction}.
            The patient asked: {prompt}
            
            Provide a helpful, professional response with medical advice appropriate for this case.
            If the bone is fractured, include first aid measures and when to see a doctor.
            If normal, suggest self-care measures and when to consult if symptoms persist.
            
            You can also mention that they can ask about specific topics like:
            - "all" for complete comprehensive guide
            - "suggestions" for treatment and recovery tips
            - "precautions" for safety measures
            - "food" for dietary recommendations
            - "medicine" for medication guidelines
            - "exercise" for physical activity recommendations
            - "emergency" for emergency signs and when to seek help
            - "recovery" for recovery timeline and progress
            """
            
            # Generate response with Gemma
            input_ids = tokenizer(context, return_tensors="pt").input_ids.to("cpu")
            response = model.generate(input_ids, max_new_tokens=150, temperature=0.7)
            bot_response = tokenizer.decode(response[0], skip_special_tokens=True)
            
            # Clean up response
            bot_response = bot_response.replace(context, "").strip()
    
    # Add bot response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    
    # Force rerun to display new messages
    st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">'
    '🩺 Medical AI Assistant | Powered by Advanced Deep Learning'
    '</div>',
    unsafe_allow_html=True
)