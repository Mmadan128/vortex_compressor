#!/bin/bash
# ============================================================================
# Vortex-Codec GraphViz Diagram Renderer
# 
# This script renders all GraphViz diagrams to multiple formats (PNG, SVG, PDF)
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if GraphViz is installed
if ! command -v dot &> /dev/null; then
    echo -e "${RED}Error: GraphViz is not installed${NC}"
    echo "Please install it:"
    echo "  Ubuntu/Debian: sudo apt install graphviz"
    echo "  macOS: brew install graphviz"
    echo "  Fedora: sudo dnf install graphviz"
    exit 1
fi

# Create output directory
OUTPUT_DIR="figures"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Rendering Vortex-Codec diagrams...${NC}\n"

# Array of diagram names and their graph names in the DOT file
declare -A DIAGRAMS=(
    ["figure1_compression_pipeline"]="compression_pipeline"
    ["figure2_model_architecture"]="model_architecture"
    ["figure3_transformer_block"]="transformer_block"
    ["figure4_compressive_attention"]="compressive_attention"
    ["figure5_training_loop"]="training_loop"
    ["figure6_data_flow"]="data_flow"
    ["figure7_parameters"]="parameters"
    ["figure8_performance"]="performance"
    ["figure9_ablation"]="ablation"
    ["figure10_arithmetic_coding"]="arithmetic_coding"
)

# Extract individual graphs from the master DOT file
DOT_FILE="diagrams_graphviz.dot"

if [ ! -f "$DOT_FILE" ]; then
    echo -e "${RED}Error: $DOT_FILE not found${NC}"
    exit 1
fi

# Function to extract a single graph
extract_graph() {
    local graph_name=$1
    local output_file=$2
    
    # Extract the graph definition
    awk "/^(di)?graph $graph_name/,/^}/" "$DOT_FILE" > "$output_file"
}

# Function to render a diagram in multiple formats
render_diagram() {
    local name=$1
    local dot_file="$OUTPUT_DIR/$name.dot"
    
    if [ ! -f "$dot_file" ]; then
        echo -e "${YELLOW}Warning: $dot_file not found, skipping${NC}"
        return
    fi
    
    echo -e "  Rendering ${GREEN}$name${NC}..."
    
    # PNG (high resolution for presentations)
    dot -Tpng -Gdpi=300 "$dot_file" -o "$OUTPUT_DIR/$name.png" 2>/dev/null
    
    # SVG (vector format for web/scaling)
    dot -Tsvg "$dot_file" -o "$OUTPUT_DIR/$name.svg" 2>/dev/null
    
    # PDF (for LaTeX papers)
    dot -Tpdf "$dot_file" -o "$OUTPUT_DIR/$name.pdf" 2>/dev/null
    
    echo -e "    ✓ PNG, SVG, PDF generated"
}

# Extract all individual graphs
echo -e "${YELLOW}Extracting individual graphs...${NC}"
for name in "${!DIAGRAMS[@]}"; do
    graph_name="${DIAGRAMS[$name]}"
    extract_graph "$graph_name" "$OUTPUT_DIR/$name.dot"
    echo "  ✓ Extracted $name"
done

echo ""

# Render all diagrams
echo -e "${YELLOW}Rendering diagrams to PNG, SVG, PDF...${NC}"
for name in "${!DIAGRAMS[@]}"; do
    render_diagram "$name"
done

echo ""
echo -e "${GREEN}✓ All diagrams rendered successfully!${NC}"
echo ""
echo "Output directory: $OUTPUT_DIR/"
echo ""
echo "Files created:"
echo "  • PNG files (300 DPI) - for presentations"
echo "  • SVG files - for web/HTML and infinite scaling"
echo "  • PDF files - for LaTeX papers"
echo ""
echo "Usage in LaTeX:"
echo '  \includegraphics[width=0.8\textwidth]{figures/figure1_compression_pipeline.pdf}'
echo ""
echo "Usage in Markdown:"
echo '  ![Architecture](figures/figure2_model_architecture.png)'
echo ""

# Generate a LaTeX include file
echo -e "${YELLOW}Generating LaTeX includes...${NC}"
cat > "$OUTPUT_DIR/figures.tex" << 'EOF'
% LaTeX figure includes for Vortex-Codec paper
% Include this file in your main document or copy relevant figures

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/figure1_compression_pipeline.pdf}
    \caption{Vortex-Codec compression pipeline showing the complete flow from input binary data through the compressive transformer to compressed output.}
    \label{fig:compression_pipeline}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/figure2_model_architecture.pdf}
    \caption{Complete model architecture with 14.8M parameters. The model consists of byte embedding, 8 compressive transformer layers, and output projection.}
    \label{fig:model_architecture}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/figure3_transformer_block.pdf}
    \caption{Detailed view of a single compressive transformer block (1.84M parameters) showing compressive attention and feed-forward components.}
    \label{fig:transformer_block}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/figure4_compressive_attention.pdf}
    \caption{Compressive attention mechanism with two-tier memory system: recent memory (512 tokens) and compressed memory (4:1 ratio).}
    \label{fig:compressive_attention}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/figure5_training_loop.pdf}
    \caption{Training loop with compressive memory management. The critical memory detachment step prevents gradient flow through sequence history.}
    \label{fig:training_loop}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/figure6_data_flow.pdf}
    \caption{Complete data flow from raw ATLAS data through preprocessing, training, and evaluation.}
    \label{fig:data_flow}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/figure7_parameters.pdf}
    \caption{Parameter distribution across model components. Transformer blocks comprise 98.2\% of the 14.8M total parameters.}
    \label{fig:parameters}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/figure8_performance.pdf}
    \caption{Performance comparison showing Vortex-Codec achieves 39\% better compression than baselines with speed trade-off.}
    \label{fig:performance}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.85\textwidth]{figures/figure9_ablation.pdf}
    \caption{Ablation study results demonstrating the importance of each architectural component. Removing memory causes 18\% degradation.}
    \label{fig:ablation}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/figure10_arithmetic_coding.pdf}
    \caption{Arithmetic coding process converting model probabilities to compressed bits achieving near-optimal compression.}
    \label{fig:arithmetic_coding}
\end{figure}
EOF

echo -e "${GREEN}✓ LaTeX includes generated: $OUTPUT_DIR/figures.tex${NC}"
echo ""

# Generate markdown gallery
cat > "$OUTPUT_DIR/README.md" << 'EOF'
# Vortex-Codec Figures

This directory contains all rendered figures for the Vortex-Codec paper.

## Figures Gallery

### Figure 1: Compression Pipeline
![Compression Pipeline](figure1_compression_pipeline.png)
Complete flow from input to compressed output.

### Figure 2: Model Architecture
![Model Architecture](figure2_model_architecture.png)
Full 14.8M parameter model structure.

### Figure 3: Transformer Block Detail
![Transformer Block](figure3_transformer_block.png)
Single block with compressive attention and feed-forward network.

### Figure 4: Compressive Attention
![Compressive Attention](figure4_compressive_attention.png)
Two-tier memory system with 4:1 compression.

### Figure 5: Training Loop
![Training Loop](figure5_training_loop.png)
Memory management during training with critical detachment step.

### Figure 6: Data Flow
![Data Flow](figure6_data_flow.png)
End-to-end data processing from ATLAS to results.

### Figure 7: Parameters Breakdown
![Parameters](figure7_parameters.png)
Distribution of 14.8M parameters across components.

### Figure 8: Performance Comparison
![Performance](figure8_performance.png)
Comparison with Gzip and Zstandard baselines.

### Figure 9: Ablation Study
![Ablation](figure9_ablation.png)
Impact of removing each architectural component.

### Figure 10: Arithmetic Coding
![Arithmetic Coding](figure10_arithmetic_coding.png)
Probability to bits conversion process.

## File Formats

Each figure is available in three formats:
- **PNG** (300 DPI) - For presentations and high-quality raster graphics
- **SVG** - For web display and infinite scaling
- **PDF** - For LaTeX papers and publication-quality printing

## Usage

### In LaTeX
```latex
\includegraphics[width=0.8\textwidth]{figures/figure1_compression_pipeline.pdf}
```

### In Markdown
```markdown
![Architecture](figure2_model_architecture.png)
```

### In HTML
```html
<img src="figures/figure2_model_architecture.svg" alt="Model Architecture">
```
EOF

echo -e "${GREEN}✓ Markdown gallery generated: $OUTPUT_DIR/README.md${NC}"
echo ""
echo -e "${GREEN}All done! 🎉${NC}"
