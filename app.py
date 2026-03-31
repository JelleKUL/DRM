import streamlit as st
import drm
from drm import pipeline
import trimesh
from trimesh.viewer import scene_to_html

st.title("3D Scene Completion")
st.text("Reconstruct a partially scanned scene")
# Sidebar
threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)

# The different workflows
tabs = st.tabs(["Preprosessing","Object Detection", "Scene Completion", "Object Completion", "Reconstruction"])

with tabs[0]:
    st.subheader("Preprocessing")
    file = st.file_uploader("Upload scene",type=["obj","ply","stl","glb","gltf"])

    model = st.selectbox("model",["votenet", "TR3D"])

    if st.button("Run object detection"):

        pcd = trimesh.load(file, file_type=file.name.split('.')[-1])
        scene = drm.pipeline.detect_objects(pcd, model)


        html = scene_to_html(trimesh.Scene(scene))

        st.components.v1.html(html, height=800)

