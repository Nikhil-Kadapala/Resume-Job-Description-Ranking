from pathlib import Path
from typing import Dict, Any, List
import json
import math
import statistics

def load_annotator_rankings(annotations_file_path: str) -> Dict[str, List[List[int]]]:
    """
    Load annotator rankings from a JSON file in the format {ranklist_i: [[rankings], ...]}.

    Args:
        annotations_file_path (str): Path to the annotations JSON file.

    Returns:
        Dict[str, List[List[int]]]: Dictionary mapping ranklist IDs to lists of ranked job indices.
    """
    if not Path(annotations_file_path).exists():
        print(f"Error: {annotations_file_path} does not exist.")
        return {}
            
    with open(annotations_file_path, 'r') as f:
        data = json.load(f)
        # Ensure only ranklist_i keys are included
        rankings = {k: v for k, v in data.items() if k.startswith("ranklist_")}
        return rankings

def get_scores(file_path: Path) -> Dict[str, List[float]]:
    """
    Get scores from a JSON file.

    Args:
        file_path (Path): Path to the JSON file.

    Returns:
        Dict[str, List[float]]: Dictionary containing the scores.
    """
    scores: Dict[str, List[float]] = {}
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f, 1):
            key = f"Resume_{idx}"
            line = json.loads(line)
            scores[key] = [item['overall_score'] for item in line[key]]
    return scores

def degree_to_numeric(degree):
    """Convert degree level to numeric value, handling variations in naming."""
    # Normalize the degree string
    try:
        degree = degree[0].lower().replace("'", "")  # Remove apostrophes (e.g., Master's → Masters)
        degree = degree[0].replace(".", "")  # Remove periods (e.g., Ph.D → PhD)
        # Handle plural forms by removing 's' if it's at the end
        if degree.endswith("s"):
            degree = degree[:-1]
    except Exception as e:
        print(f"Error in degree_to_numeric: {e}, {degree}")
    
    # Standardize "phd" variations
    if degree in ["phd", "ph d"]:
        degree = "phd"
    
    # Mapping of normalized degree names
    mapping = {
        "high school": 1,
        "associate": 2,
        "bachelor": 3,
        "master": 4,
        "phd": 5
    }
    return mapping.get(degree, 0)

def degree_sim(c_degree, r_degree):
    """Compute DegreeSim(c,r) based on degree levels.
    
    Args:
        c_degree: Candidate's degree, can be string or list of strings
        r_degree: Required degree, can be string or list of strings
                  If empty list, returns 1 (no degree requirement)
    """
    # If required degree is None, empty string, or empty list, return 1 (no requirement)
    if not r_degree or (isinstance(r_degree, list) and len(r_degree) == 0):
        return 1

    if not c_degree or (isinstance(c_degree, list) and len(c_degree) == 0):
        return 0
        
    # Handle case where degrees are in lists
    if isinstance(c_degree, list) and len(c_degree) > 0:
        c_degree = c_degree[0] if c_degree else ""
    if isinstance(r_degree, list) and len(r_degree) > 0:
        r_degree = r_degree[0] if r_degree else ""
    
    try:
        cand_level = degree_to_numeric(c_degree)
        req_level = degree_to_numeric(r_degree)
        
        if cand_level < req_level:
            return 0
        diff = cand_level - req_level
        if 0 <= diff < 2:
            return 1
        return 0.5
    except Exception as e:
        print(f"Error in degree_sim: {e}")
        return 0

def majors_sim(r_major, j_major):
    """Compute MajorsSim(r,j) based on major similarity.
    
    Args:
        r_major: Required major, can be string or list of strings
                If empty list, returns 1 (no major requirement)
        j_major: Job's major, can be string or list of strings
    """
    # If required major is None, empty string, or empty list, return 1 (no requirement)
    if not j_major or (isinstance(j_major, list) and len(j_major) == 0):
        return 1
    
    if not r_major or (isinstance(r_major, list) and len(r_major) == 0):
        return 0

    if isinstance(j_major, list) and len(j_major) > 0:
        j_major = j_major[0] if j_major else ""

    if isinstance(r_major, list) and len(r_major) > 0:
        r_major = r_major[0] if r_major else ""
    
    related_majors = {
        "Computer Science": ["Computer Science", "Information Technology", "Cyber Security", "Data Engineering", "Data Science", "Artificial Intelligence", "Machine Learning", "Deep Learning", "Computer Engineering", "Software Engineering"],
        "Statistics": ["Statistics", "Mathematics", "Data Science", "Data Analysis", "Business Analytics", "Business Intelligence", "Finance", "Economics", "Marketing", "Operations Research", "Supply Chain Management"],
        "Decision Science": ["Decision Science", "Mathematics", "Statistics", "Data Science", "Data Analysis", "Business Analytics", "Business Intelligence", "Finance", "Economics", "Marketing", "Operations Research", "Supply Chain Management"],
        "Mathematics": ["Data Science", "Physics", "Applied Mathematics", "Statistics"],
        "Electrical Engineering": ["Electrical Engineering", "Electronics", "Computer Engineering", "VLSI Engineering", "Communication Engineering", "Electronics and Communication Engineering", "Electronics and Computer Engineering", "Electronics and Electrical Engineering"],
    }
    
    try:
        r_major = r_major.lower()
        j_major = j_major.lower()
    except Exception as e:
        print(f"Error: {e}, {r_major}, {j_major}")
    
    if r_major == j_major:
        return 1
    related_set = related_majors.get(j_major, [])
    if r_major in [m.lower() for m in related_set]:
        return 0.5
    return 0

def get_cdegree_rdegree(filepath: Path) -> Dict[str, Dict[str, List[Dict[str, List[str]]]]]:
    cdegree_rdegree_dict: Dict[str, Dict[str, List[Dict[str, List[str]]]]] = {}
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f, 1):
            key = f"Resume_{idx}"
            line = json.loads(line)
            c_edu = [
                {
                    "degree": item.get('resume', {}).get('qualifications', {}).get('EDUCATION', {}).get("degree", []),
                    "level": item.get('resume', {}).get('qualifications', {}).get('EDUCATION', {}).get("level", []),
                    "major": item.get('resume', {}).get('qualifications', {}).get('EDUCATION', {}).get("major", [])
                }
                for item in line[key]
            ]
            r_edu = [
                {
                    "required_degree": item.get('job_description', {}).get('EDUCATION', {}).get("required_degree", []),
                    "preferred_degree": item.get('job_description', {}).get('EDUCATION', {}).get("preferred_degree", []),
                    "required_level": item.get('job_description', {}).get('EDUCATION', {}).get("required_level", []),
                    "preferred_level": item.get('job_description', {}).get('EDUCATION', {}).get("preferred_level", []),
                    "required_major": item.get('job_description', {}).get('EDUCATION', {}).get("required_major", []),
                    "preferred_major": item.get('job_description', {}).get('EDUCATION', {}).get("preferred_major", [])
                }
                for item in line[key]
            ]
            cdegree_rdegree_dict[key] = {"resume_education": c_edu, "job_education": r_edu}
    return cdegree_rdegree_dict

def compute_adjusted_scores(scores: List[float], resume_education: List[Dict[str, List[str]]], job_education: List[Dict[str, List[str]]]) -> List[float]:
    """
    Compute adjusted scores by resolving duplicates using DegreeSim and MajorsSim,
    adding a ranking factor for uniqueness, and ensuring adjusted scores don't exceed
    the next highest original score.

    Args:
        scores (List[float]): List of overall scores for jobs.
        resume_education (List[Dict[str, List[str]]]): Candidate's education for each job.
        job_education (List[Dict[str, List[str]]]): Job education requirements for each job.

    Returns:
        List[float]: Adjusted scores for each job.
    """
    from collections import defaultdict

    # Step 1: Identify duplicate scores and find caps
    score_groups = defaultdict(list)
    for idx, score in enumerate(scores):
        score_groups[score].append(idx)
    
    # Get sorted unique scores to determine caps
    unique_scores = sorted(set(scores))
    
    # Map each score to the next highest score (cap)
    score_caps = {}
    for i, score in enumerate(unique_scores):
        if i + 1 < len(unique_scores):
            score_caps[score] = unique_scores[i + 1]
        else:
            score_caps[score] = float('inf')  # No cap for the highest score
    
    # Step 2: Compute adjusted scores with caps
    adjusted_scores = scores.copy()
    rank_factor = 0.01  # Small factor to ensure uniqueness within adjustments
    
    for score, indices in score_groups.items():
        if len(indices) > 1:  # Only adjust if there are duplicates
            # Compute DegreeSim and MajorsSim for each job
            adjustments = []
            for idx in indices:
                degree_score = degree_sim(resume_education[idx]["level"], job_education[idx]["required_degree"])
                majors_score = majors_sim(resume_education[idx]["major"], job_education[idx]["required_major"])
                total_adjustment = degree_score + majors_score
                adjustments.append((idx, total_adjustment))
            
            # Sort by adjustment (descending) to prioritize better matches
            adjustments.sort(key=lambda x: x[1], reverse=True)
            
            # Determine the cap for this score
            cap = score_caps[score]
            
            # Apply adjustments, ensuring they don't exceed the cap
            for rank, (idx, adjustment) in enumerate(adjustments):
                # Base adjustment plus rank factor for uniqueness
                adjusted = score + adjustment * 0.01 + (len(indices) - rank - 1) * rank_factor
                # Cap the adjusted score to be less than the next highest score
                if cap != float('inf'):
                    adjusted = min(adjusted, cap - rank_factor)
                adjusted_scores[idx] = adjusted
    
    return adjusted_scores

def compute_kendall_tau(model_ranking: List[int], annotator_ranking: List[int]) -> float:
    """
    Compute Kendall's Tau-b correlation coefficient between two ranked lists.

    Args:
        model_ranking (List[int]): Model's ranked job indices (1-based).
        annotator_ranking (List[int]): Annotator's ranked job indices (1-based).

    Returns:
        float: Kendall's Tau-b coefficient (-1 to 1), or 0.0 for invalid inputs.
    """
    if not model_ranking or not annotator_ranking or len(model_ranking) != len(annotator_ranking):
        print("Error: Invalid or mismatched ranking lists.")
        return 0.0

    n = len(model_ranking)
    concordant = 0
    discordant = 0
    model_ties = 0
    annotator_ties = 0

    # Compare all pairs
    for i in range(n):
        for j in range(i + 1, n):
            model_diff = model_ranking[i] - model_ranking[j]
            annotator_diff = annotator_ranking[i] - annotator_ranking[j]
            
            # Concordant: same relative order
            if model_diff * annotator_diff > 0:
                concordant += 1
            # Discordant: opposite relative order
            elif model_diff * annotator_diff < 0:
                discordant += 1
            # Ties: count ties in each ranking
            else:
                if model_diff == 0:
                    model_ties += 1
                if annotator_diff == 0:
                    annotator_ties += 1

    # Total pairs
    total_pairs = n * (n - 1) / 2
    # Adjust for ties
    denominator = math.sqrt((total_pairs - model_ties) * (total_pairs - annotator_ties))
    
    if denominator == 0:
        return 0.0  # Avoid division by zero
    
    # Tau-b formula
    tau = (concordant - discordant) / denominator
    return tau

curr_dir = Path(__file__).parent
parent_dir = curr_dir.parent.parent
INFERENCE_FILE_PATH = parent_dir / "data" / "results" / "inference" / "inference.jsonl"
ANNOTATIONS_FILE_PATH = parent_dir / "data" / "test" / "annotations.json"
scores_list = get_scores(INFERENCE_FILE_PATH)
cdegree_rdegree_dict = get_cdegree_rdegree(INFERENCE_FILE_PATH)
annotator_rankings = load_annotator_rankings(ANNOTATIONS_FILE_PATH)

adjusted_scores: Dict[str, List[float]] = {}
ranked_indices: Dict[str, List[int]] = {}
alignment_scores: Dict[str, Dict[str, float]] = {}
print(scores_list)
print("==========================================================")
# Limit to first 10 resumes
for idx, (k, v) in enumerate(list(scores_list.items())[:10], 1):
    # Get education data for all jobs of this resume
    resume_edu = cdegree_rdegree_dict[k]['resume_education']
    job_edu = cdegree_rdegree_dict[k]['job_education']
    # Compute adjusted scores for all jobs
    adjusted_scores[k] = compute_adjusted_scores(v, resume_edu, job_edu)
    # Rank adjusted scores in descending order and get 1-based indices
    indexed_scores = [(score, idx) for idx, score in enumerate(adjusted_scores[k])]
    indexed_scores.sort(key=lambda x: (-x[0], x[1]))  # Sort by score (descending), then index (ascending)
    ranked_indices[k] = [idx + 1 for _, idx in indexed_scores]  # Convert to 1-based indices
    # Compute alignment with annotator rankings for each ranklist (first 10 resumes)
    alignment_scores[k] = {}
    for ranklist_id, rankings in annotator_rankings.items():
        if idx - 1 < len(rankings[:10]):  # Limit to first 10 rankings
            annotator_ranking = rankings[idx - 1]
            alignment_scores[k][ranklist_id] = compute_kendall_tau(ranked_indices[k], annotator_ranking)
        else:
            print(f"Warning: No annotator ranking found for {k} in {ranklist_id}")
            alignment_scores[k][ranklist_id] = 0.0

# Compute overall performance metrics
per_resume_mean_tau = {}
per_ranklist_mean_tau = {ranklist_id: [] for ranklist_id in annotator_rankings}
inter_annotator_tau = []
all_tau_values = []

# Calculate per-resume mean Tau and collect Tau values
for k in alignment_scores:
    tau_values = [tau for tau in alignment_scores[k].values() if tau != 0.0]  # Exclude missing rankings
    if tau_values:
        per_resume_mean_tau[k] = statistics.mean(tau_values)
        all_tau_values.extend(tau_values)
    else:
        per_resume_mean_tau[k] = 0.0
    # Collect Tau for per-ranklist means
    for ranklist_id in per_ranklist_mean_tau:
        if ranklist_id in alignment_scores[k] and alignment_scores[k][ranklist_id] != 0.0:
            per_ranklist_mean_tau[ranklist_id].append(alignment_scores[k][ranklist_id])

# Compute inter-annotator agreement (Tau between ranklist_1 and ranklist_2)
if "ranklist_1" in annotator_rankings and "ranklist_2" in annotator_rankings:
    for idx in range(min(10, len(annotator_rankings["ranklist_1"]), len(annotator_rankings["ranklist_2"]))):
        tau = compute_kendall_tau(annotator_rankings["ranklist_1"][idx], annotator_rankings["ranklist_2"][idx])
        if tau != 0.0:  # Exclude invalid comparisons
            inter_annotator_tau.append(tau)

# Compute overall metrics
overall_mean_tau = statistics.mean(all_tau_values) if all_tau_values else 0.0
overall_std_tau = statistics.stdev(all_tau_values) if len(all_tau_values) > 1 else 0.0
per_ranklist_means = {
    ranklist_id: statistics.mean(taus) if taus else 0.0
    for ranklist_id, taus in per_ranklist_mean_tau.items()
}
inter_annotator_mean_tau = statistics.mean(inter_annotator_tau) if inter_annotator_tau else 0.0

# Print results
print("Adjusted Scores:", adjusted_scores)
print("Ranked Indices:", ranked_indices)
print("Alignment Scores (Kendall's Tau):", alignment_scores)
print("==========================================================")
print("Performance Metrics:")
print(f"Overall Mean Kendall's Tau: {overall_mean_tau:.3f}")
print(f"Overall Standard Deviation of Tau: {overall_std_tau:.3f}")
print("Per-Resume Mean Kendall's Tau:")
for k, mean_tau in per_resume_mean_tau.items():
    print(f"  {k}: {mean_tau:.3f}")
print("Per-Ranklist Mean Kendall's Tau:")
for ranklist_id, mean_tau in per_ranklist_means.items():
    print(f"  {ranklist_id}: {mean_tau:.3f}")
print(f"Inter-Annotator Mean Kendall's Tau: {inter_annotator_mean_tau:.3f}")