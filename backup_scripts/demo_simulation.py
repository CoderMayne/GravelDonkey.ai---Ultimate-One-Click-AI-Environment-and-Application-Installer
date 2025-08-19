#!/usr/bin/env python3
"""
GPU Simulation Demo for AI Installer

This script demonstrates how to integrate GPU simulation into the existing
installer app for testing AMD and Intel GPU detection without actual hardware.
"""

import sys
import os
import gradio as gr

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_hardware_detector import get_gpu_info, get_recommended_dependency

def create_simulation_demo():
    """Create a Gradio demo for GPU simulation testing."""
    
    def simulate_gpu_detection(gpu_type, gpu_model):
        """Simulate GPU detection based on user selection."""
        if gpu_type == "Real Hardware":
            simulation_mode = "auto"
            gpu_type_param = "amd"  # Default, will be ignored in auto mode
            gpu_model_param = "rx_7900_xtx"  # Default, will be ignored in auto mode
        elif gpu_type == "AMD":
            simulation_mode = "force_amd"
            gpu_type_param = "amd"
            gpu_model_param = gpu_model
        elif gpu_type == "Intel":
            simulation_mode = "force_intel"
            gpu_type_param = "intel"
            gpu_model_param = gpu_model
        else:
            return "Invalid GPU type selected"
        
        # Get GPU information
        gpu_info = get_gpu_info(simulation_mode, gpu_type_param, gpu_model_param)
        
        # Format the results
        result = f"""
## GPU Detection Results

**GPU Type:** {gpu_info[0].upper()}
**GPU Model:** {gpu_info[1]}
**Architecture:** {gpu_info[6]}
**Compute Capability:** {gpu_info[7]}
"""
        
        if gpu_info[4]:  # Memory total
            result += f"**Total Memory:** {gpu_info[4]} MB\n"
        if gpu_info[5]:  # Memory used
            result += f"**Used Memory:** {gpu_info[5]} MB\n"
        if gpu_info[2]:  # CUDA version
            result += f"**CUDA Version:** {gpu_info[2]}\n"
        if gpu_info[3]:  # Driver version
            result += f"**Driver Version:** {gpu_info[3]}\n"
        
        # Get recommended dependencies
        recommended = get_recommended_dependency(gpu_info[0], gpu_info[1], gpu_info[6], gpu_info[4])
        result += f"\n**Recommended Configuration:** {recommended}\n"
        
        # Add simulation notice if applicable
        if simulation_mode != "auto":
            result += f"\n⚠️ **SIMULATION MODE:** This is simulated data for testing purposes.\n"
        
        return result
    
    def get_available_models(gpu_type):
        """Get available models for the selected GPU type."""
        if gpu_type == "AMD":
            return gr.Dropdown(
                choices=[
                    "rx_7900_xtx (24GB, RDNA 3)",
                    "rx_7800_xt (16GB, RDNA 3)", 
                    "rx_6800_xt (16GB, RDNA 2)",
                    "rx_6700_xt (12GB, RDNA 2)",
                    "rx_5700_xt (8GB, RDNA)"
                ],
                value="rx_7900_xtx (24GB, RDNA 3)",
                label="AMD GPU Model"
            )
        elif gpu_type == "Intel":
            return gr.Dropdown(
                choices=[
                    "arc_a770 (16GB, Xe-HPG)",
                    "arc_a750 (8GB, Xe-HPG)",
                    "arc_a580 (8GB, Xe-HPG)",
                    "arc_a380 (6GB, Xe-HPG)",
                    "iris_xe (4GB, Xe-LP)"
                ],
                value="arc_a770 (16GB, Xe-HPG)",
                label="Intel GPU Model"
            )
        else:
            return gr.Dropdown(choices=[], label="GPU Model")
    
    # Create the interface
    with gr.Blocks(title="GPU Simulation Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎮 GPU Simulation Demo")
        gr.Markdown("""
        This demo allows you to test GPU detection for AMD and Intel GPUs without requiring actual hardware.
        
        **Features:**
        - Simulate AMD Radeon RX series (RDNA 3, RDNA 2, RDNA)
        - Simulate Intel Arc series and Iris Xe graphics
        - Test real hardware detection (if available)
        - Get recommended dependency configurations
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")
                
                gpu_type_selector = gr.Dropdown(
                    choices=[
                        "Real Hardware",
                        "AMD", 
                        "Intel"
                    ],
                    value="AMD",
                    label="GPU Type"
                )
                
                gpu_model_selector = gr.Dropdown(
                    choices=[
                        "rx_7900_xtx (24GB, RDNA 3)",
                        "rx_7800_xt (16GB, RDNA 3)", 
                        "rx_6800_xt (16GB, RDNA 2)",
                        "rx_6700_xt (12GB, RDNA 2)",
                        "rx_5700_xt (8GB, RDNA)"
                    ],
                    value="rx_7900_xtx (24GB, RDNA 3)",
                    label="GPU Model"
                )
                
                detect_btn = gr.Button("🔍 Detect GPU", variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                results_output = gr.Markdown("Select a GPU type and model, then click 'Detect GPU' to see the results.")
        
        # Event handlers
        gpu_type_selector.change(
            fn=get_available_models,
            inputs=[gpu_type_selector],
            outputs=[gpu_model_selector]
        )
        
        detect_btn.click(
            fn=simulate_gpu_detection,
            inputs=[gpu_type_selector, gpu_model_selector],
            outputs=[results_output]
        )
        
        # Add some helpful information
        with gr.Accordion("ℹ️ About GPU Simulation", open=False):
            gr.Markdown("""
            **What is GPU Simulation?**
            
            GPU simulation allows developers and testers to simulate different GPU configurations
            without requiring actual hardware. This is useful for:
            
            - Testing installer functionality with different GPU types
            - Developing and debugging GPU-specific features
            - Demonstrating capabilities to users without specific hardware
            - CI/CD testing across different GPU configurations
            
            **Supported GPU Types:**
            
            **AMD Radeon Series:**
            - RX 7900 XTX (24GB, RDNA 3) - Latest flagship
            - RX 7800 XT (16GB, RDNA 3) - High-end gaming
            - RX 6800 XT (16GB, RDNA 2) - Previous generation flagship
            - RX 6700 XT (12GB, RDNA 2) - Mid-range gaming
            - RX 5700 XT (8GB, RDNA) - Previous generation mid-range
            
            **Intel Arc Series:**
            - Arc A770 (16GB, Xe-HPG) - Flagship discrete GPU
            - Arc A750 (8GB, Xe-HPG) - High-end discrete GPU
            - Arc A580 (8GB, Xe-HPG) - Mid-range discrete GPU
            - Arc A380 (6GB, Xe-HPG) - Entry-level discrete GPU
            - Iris Xe (4GB, Xe-LP) - Integrated graphics
            
            **Real Hardware Detection:**
            - Automatically detects NVIDIA, AMD, or Intel GPUs if present
            - Falls back to CPU if no GPU is detected
            - Provides accurate hardware information when available
            """)
    
    return demo

def main():
    """Main function to run the demo."""
    print("Starting GPU Simulation Demo...")
    
    # Create and launch the demo
    demo = create_simulation_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
