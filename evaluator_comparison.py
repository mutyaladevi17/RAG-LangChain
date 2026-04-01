import gradio as gr
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv

from evaluation.eval import evaluate_all_retrieval, evaluate_all_answers
from implementation.answer import get_vector_store_name

load_dotenv(override=True)

# Color coding thresholds - Retrieval
MRR_GREEN = 0.9
MRR_AMBER = 0.75
NDCG_GREEN = 0.9
NDCG_AMBER = 0.75
COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0

# Color coding thresholds - Answer (1-5 scale)
ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0


def get_color(value: float, metric_type: str) -> str:
    """Get color based on metric value and type."""
    if metric_type == "mrr":
        if value >= MRR_GREEN:
            return "green"
        elif value >= MRR_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "ndcg":
        if value >= NDCG_GREEN:
            return "green"
        elif value >= NDCG_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "coverage":
        if value >= COVERAGE_GREEN:
            return "green"
        elif value >= COVERAGE_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type in ["accuracy", "completeness", "relevance"]:
        if value >= ANSWER_GREEN:
            return "green"
        elif value >= ANSWER_AMBER:
            return "orange"
        else:
            return "red"
    return "black"


def format_metric_html(
    label: str,
    value: float,
    metric_type: str,
    is_percentage: bool = False,
    score_format: bool = False,
) -> str:
    """Format a metric with color coding."""
    color = get_color(value, metric_type)
    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value_str}</div>
    </div>
    """


def run_retrieval_evaluation_single(db_type: str, k: int, progress=gr.Progress()):
    """Run retrieval evaluation for a single configuration."""
    total_mrr = 0.0
    total_ndcg = 0.0
    total_coverage = 0.0
    category_mrr = defaultdict(list)
    count = 0

    get_vector_store_name(db_type=db_type, retriver_k=k)

    for test, result, prog_value in evaluate_all_retrieval():
        count += 1
        total_mrr += result.mrr
        total_ndcg += result.ndcg
        total_coverage += result.keyword_coverage

        category_mrr[test.category].append(result.mrr)

        # Update progress bar only
        progress(prog_value, desc=f"Evaluating {db_type} (k={k}) test {count}...")

    # Calculate final averages
    if count == 0:
        return "<div>No tests found</div>", pd.DataFrame()

    avg_mrr = total_mrr / count
    avg_ndcg = total_ndcg / count
    avg_coverage = total_coverage / count

    # Create final summary metrics HTML
    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Mean Reciprocal Rank (MRR)", avg_mrr, "mrr")}
        {format_metric_html("Normalized DCG (nDCG)", avg_ndcg, "ndcg")}
        {format_metric_html("Keyword Coverage", avg_coverage, "coverage", is_percentage=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Complete: {count} tests</span>
        </div>
    </div>
    """

    # Create final bar chart data
    category_data = []
    for category, mrr_scores in category_mrr.items():
        avg_cat_mrr = sum(mrr_scores) / len(mrr_scores)
        category_data.append({"Category": category, "Average MRR": avg_cat_mrr})

    df = pd.DataFrame(category_data)
    return final_html, df


def run_all_retrieval_cases(progress=gr.Progress()):
    """Run retrieval evaluation for all db configurations sequentially."""
    configs = [("small", 10), ("large", 5), ("hybrid", 7)]
    results = []
    
    for db_type, k in configs:
        html, df = run_retrieval_evaluation_single(db_type, k, progress)
        results.extend([html, df])
        
    return tuple(results)


def run_answer_evaluation_single(db_type: str, k: int, progress=gr.Progress()):
    """Run answer evaluation for a single configuration."""
    total_accuracy = 0.0
    total_completeness = 0.0
    total_relevance = 0.0
    category_accuracy = defaultdict(list)
    count = 0

    get_vector_store_name(db_type=db_type, retriver_k=k)

    for test, result, prog_value in evaluate_all_answers():
        count += 1
        total_accuracy += result.accuracy
        total_completeness += result.completeness
        total_relevance += result.relevance

        category_accuracy[test.category].append(result.accuracy)

        # Update progress bar only
        progress(prog_value, desc=f"Evaluating {db_type} (k={k}) test {count}...")

    if count == 0:
        return "<div>No tests found</div>", pd.DataFrame()

    # Calculate final averages
    avg_accuracy = total_accuracy / count
    avg_completeness = total_completeness / count
    avg_relevance = total_relevance / count

    # Create final summary metrics HTML
    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Accuracy", avg_accuracy, "accuracy", score_format=True)}
        {format_metric_html("Completeness", avg_completeness, "completeness", score_format=True)}
        {format_metric_html("Relevance", avg_relevance, "relevance", score_format=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Complete: {count} tests</span>
        </div>
    </div>
    """

    # Create final bar chart data
    category_data = []
    for category, accuracy_scores in category_accuracy.items():
        avg_cat_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        category_data.append({"Category": category, "Average Accuracy": avg_cat_accuracy})

    df = pd.DataFrame(category_data)
    return final_html, df


def run_all_answer_cases(progress=gr.Progress()):
    """Run answer evaluation for all db configurations sequentially."""
    configs = [("small", 10), ("large", 5), ("hybrid", 7)]
    results = []
    
    for db_type, k in configs:
        html, df = run_answer_evaluation_single(db_type, k, progress)
        results.extend([html, df])
        
    return tuple(results)


def main():
    """Launch the Gradio evaluation app."""
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Evaluation Dashboard", theme=theme) as app:
        gr.Markdown("# 📊 RAG Evaluation Dashboard - Comparison")
        gr.Markdown("Evaluate retrieval and answer quality across different vectorstore configurations.")

        configs = [("small", 10), ("large", 5), ("hybrid", 7)]

        # RETRIEVAL SECTION
        gr.Markdown("## 🔍 Retrieval Evaluation Comparison")

        retrieval_button = gr.Button("Run All Retrieval Evaluations", variant="primary", size="lg")
        
        retrieval_outputs = []
        with gr.Row():
            for db_type, k in configs:
                with gr.Column(scale=1):
                    gr.Markdown(f"### {db_type.capitalize()} DB (k={k})")
                    metrics = gr.HTML(
                        "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run All Retrieval Evaluations' to start</div>"
                    )
                    chart = gr.BarPlot(
                        x="Category",
                        y="Average MRR",
                        title=f"{db_type.capitalize()} MRR",
                        y_lim=[0, 1],
                        height=300,
                    )
                    retrieval_outputs.extend([metrics, chart])

        # ANSWERING SECTION
        gr.Markdown("## 💬 Answer Evaluation Comparison")

        answer_button = gr.Button("Run All Answer Evaluations", variant="primary", size="lg")
        
        answer_outputs = []
        with gr.Row():
            for db_type, k in configs:
                with gr.Column(scale=1):
                    gr.Markdown(f"### {db_type.capitalize()} DB (k={k})")
                    metrics = gr.HTML(
                        "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run All Answer Evaluations' to start</div>"
                    )
                    chart = gr.BarPlot(
                        x="Category",
                        y="Average Accuracy",
                        title=f"{db_type.capitalize()} Accuracy",
                        y_lim=[1, 5],
                        height=300,
                    )
                    answer_outputs.extend([metrics, chart])

        # Wire up the evaluations
        retrieval_button.click(
            fn=run_all_retrieval_cases,
            outputs=retrieval_outputs,
        )

        answer_button.click(
            fn=run_all_answer_cases,
            outputs=answer_outputs,
        )

    app.launch(inbrowser=True)


if __name__ == "__main__":
    main()
