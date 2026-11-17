
import streamlit as st
from PIL import Image
from io import BytesIO

from style_transfer import load_models, generate_stylized_image


# Page configuration
st.set_page_config(
    page_title="AdaIN Neural Style Transfer",
    page_icon="🎨",
    layout="wide"
)


# Load models only once
@st.cache_resource
def get_models():
    return load_models()


# Heading
st.title("🎨 Neural Style Transfer with AdaIN")

st.write(
    "Transform your photographs into artistic images "
    "using Adaptive Instance Normalization."
)

st.divider()


# Image upload section
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Content Image")

    content_file = st.file_uploader(
        "Upload content image",
        type=["jpg", "jpeg", "png"],
        key="content"
    )

    if content_file:
        content_image = Image.open(content_file).convert("RGB")
        st.image(
            content_image,
            caption="Content Image",
            width="stretch"
        )

with col2:
    st.subheader("🖌️ Style Image")

    style_file = st.file_uploader(
        "Upload style image",
        type=["jpg", "jpeg", "png"],
        key="style"
    )

    if style_file:
        style_image = Image.open(style_file).convert("RGB")
        st.image(
            style_image,
            caption="Style Image",
            width="stretch"
        )


st.divider()


# Style settings

# Sidebar settings
with st.sidebar:

    st.title("⚙️ Style Settings")

    st.write("Customize your style transfer.")

    st.divider()

    # Alpha slider
    st.subheader("Style Strength")

    alpha = st.slider(
        "Adjust Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    st.caption(f"Selected Alpha: {alpha}")

    st.divider()

    # Resolution selector
    st.subheader("Output Resolution")

    resolution = st.selectbox(
        "Choose image resolution",
        options=[256, 512, 768],
        index=0,
        format_func=lambda x: f"{x}px"
    )

    st.divider()

    st.info(
        "Higher resolution requires more processing time."
    )

    st.divider()

    # About Project
    with st.expander("ℹ️ About This Project"):

        st.write(
            "Neural Style Transfer with AdaIN "
            "is an AI-powered application that "
            "combines the content of one image "
            "with the artistic style of another."
        )

        st.write("**Technologies Used:**")

        st.markdown("""
        - Python
        - PyTorch
        - Torchvision
        - VGG-19
        - Adaptive Instance Normalization
        - Streamlit
        """)

        st.write("**How it works:**")

        st.write(
            "VGG-19 extracts image features, "
            "AdaIN transfers the style statistics, "
            "and a pretrained decoder generates "
            "the stylized image."
        )



# Generate button
if st.button("🎨 Generate Stylized Image", width="stretch"):

    if content_file is None or style_file is None:

        st.warning("Please upload both images.")

    else:

        with st.spinner("Generating your artistic image..."):

            try:
                content_image = Image.open(
                    content_file
                ).convert("RGB")

                style_image = Image.open(
                    style_file
                ).convert("RGB")

                encoder, decoder = get_models()

                output_image = generate_stylized_image(
                    content_image,
                    style_image,
                    alpha,
                    resolution,
                    encoder,
                    decoder
                )

                st.success("Style transfer completed!")

                st.subheader("✨ Before & After Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        content_image,
                        caption="Original Content Image",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        output_image,
                        caption="Generated Stylized Image",
                        use_container_width=True
                    )

                # Download generated image

                # Prepare generated image for download
                buffer = BytesIO()
                output_image.save(buffer, format="PNG")

                image_data = buffer.getvalue()

                # Output Information Panel
                st.divider()
                st.subheader("📊 Output Information")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Width", f"{output_image.width}px")

                with col2:
                    st.metric("Height", f"{output_image.height}px")

                with col3:
                    st.metric("Alpha", f"{alpha:.1f}")

                with col4:
                    file_size = len(image_data) / 1024
                    st.metric("File Size", f"{file_size:.2f} KB")

                st.caption(f"Selected Resolution: {resolution}px")

                # Download button
                st.download_button(
                    label="⬇️ Download Stylized Image",
                    data=image_data,
                    file_name="stylized_image.png",
                    mime="image/png",
                    width="stretch"
                )


            except Exception as error:
                st.error(f"Generation failed: {error}")


# Application Footer

st.divider()

st.markdown(
    """
<div style="text-align: center; color: gray; padding: 20px;">
    <h4>🎨 Neural Style Transfer with AdaIN</h4>
    <p>
        Powered by PyTorch, VGG-19 and
        Adaptive Instance Normalization
    </p>
    <p>AI & Machine Learning Project</p>
</div>
    """,
    unsafe_allow_html=True
)
