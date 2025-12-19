# Capstone Project – Intelligent Log Parsing & Labeling System

## Overview
This project implements an **end-to-end intelligent log parsing and labeling system** designed for large-scale system logs.  
It combines **deep learning**, **template-based parsing**, and **rule-based labeling**, and exposes the full pipeline through an **API with automated file management** for near real-time monitoring and analysis.

The system is designed to:
- Parse unstructured system logs into structured formats
- Automatically generate log templates
- Assign semantic labels to log events
- Maintain both historical and latest results automatically

---

## System Architecture
The pipeline consists of four main components:

1. **Training Pipeline**
2. **Combined Model (T5 + Drain3 + Label Rules)**
3. **Model Wrapper & API**
4. **Automated File Management System**

---

## 1. Training Process

### Dataset
The system is trained and evaluated using LogHub-style system logs, including:
- Apache
- BGL
- HDFS
- Hadoop
- HPC
- Linux
- OpenStack
- OpenSSH
- Spark
- Thunderbird
- Zookeeper

Raw log files are first **preprocessed into structured CSV format**, extracting:
- Timestamp
- Component
- Log message
- Ground-truth labels (when available)

---

### Model Training
The training phase focuses on **log template generation**.

#### Model Used
- **T5 (Text-to-Text Transfer Transformer)**

#### Training Objective
- Input: Raw log message  
- Output: Generalized log template with variables abstracted  

Example:
```
Input : Connection to 192.168.1.10 failed after 3 retries
Output: Connection to <*> failed after <*> retries
```

The T5 model learns generalized patterns across heterogeneous log formats, allowing it to generalize to unseen logs.

---

### Clustering with Drain3
After template generation:
- **Drain3** clusters similar log messages
- Each cluster is assigned a unique cluster ID
- Cluster templates enable efficient grouping and analysis

---

### Label Assignment
Semantic labels are assigned using **rule-based template mapping**:
- Templates are mapped to labels such as *INFO*, *WARNING*, *ERROR*, and *SECURITY*
- Rules are stored as JSON configuration files
- This ensures explainability and easy extensibility

---

## 2. Combined Model

The final system is packaged as a **combined model**, consisting of:
- Trained T5 model
- Drain3 template miner
- Template-to-label mappings
- Label metadata and cluster mappings

### Model Storage
Due to GitHub file size limits, the trained model is provided as a **compressed archive**:

```
backend/config/combined_model_full.rar
```

After extraction, the following files must exist:

```
backend/config/
├── combined_model_full.pkl
├── template_label_map.json
├── label_rules.json
├── label_meta_map.json
├── cluster_label_map.json
```

> The raw `.pkl` file is excluded from Git history and provided only in compressed form.

---

## 3. Model Wrapper

The **model wrapper** abstracts the combined model and provides a unified inference interface.

### Responsibilities
- Load the combined model
- Handle CPU / GPU inference
- Parse incoming logs
- Generate templates and cluster IDs
- Assign semantic labels
- Output structured results

This design ensures clean separation between **model logic** and **API logic**.

---

## 4. API Layer

The system exposes functionality via a **FastAPI-based backend**.

### Core Features
- Upload log files via HTTP
- Process logs using the combined model
- Return structured CSV / JSON outputs
- Support integration with visualization tools such as Grafana

---

## 5. Automated File Management System

The system implements an **automated file management mechanism** to support continuous operation.

### Dual-Output Strategy
For each processed log file:
1. **Archived Output**
   - Immutable historical records
   - Enables auditing and debugging

2. **Master Output**
   - Represents the latest consolidated system state
   - Automatically overwritten on each run

This design separates **historical accountability** from **real-time usability**.

---

## 6. How to Run the System

### Prerequisites
- Python 3.10+
- Conda or virtual environment
- Git LFS installed
- GPU recommended (CUDA-enabled), but CPU is supported

---

### Setup
```bash
git clone https://github.com/ilovecarbss/capstone.git
cd capstone
```

Extract the model:
```bash
cd backend/config
# Extract combined_model_full.rar
```

Install dependencies:
```bash
pip install -r backend/requirements.txt
```

---

### Start the API (Windows)
The API is started using the provided **batch file**.

```bat
start_api.bat
```

This script:
- Activates the environment
- Loads the combined model
- Starts the FastAPI server
- Initializes automated file management

---

### Usage Flow
1. Place or upload a log file
2. The API processes logs via the model wrapper
3. Results are:
   - Archived for historical reference
   - Written to the master output file
4. Outputs are ready for visualization or further analysis

---

## 7. Results
Processed outputs are stored in the `Results/` directory and include:
- Structured log entries
- Generated templates
- Cluster IDs
- Assigned semantic labels

These outputs are ready for:
- Grafana dashboards
- Offline inspection
- Evaluation and benchmarking

---

## Notes
- Large datasets are excluded from the repository
- The architecture is modular and extensible
- The system supports near real-time detection through incremental processing

---

## Conclusion
This project demonstrates a practical and scalable approach to intelligent log parsing by combining deep learning, template mining, and automated system design. The proposed framework is adaptable to real-world environments and provides a strong foundation for future research and deployment.
