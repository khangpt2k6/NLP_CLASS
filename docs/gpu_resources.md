# GPU options for the OWL project (checked 2026-10-16)

A 7B model in fp16 is about 13.5 GB. Pruning runs layer by layer, so one 16 GB GPU is enough for Wanda and OWL. SparseGPT and 13B models are more comfortable on 24 GB or more.

## USF (free)

### GAIVI cluster (CSE / Bellini College) - best option
- Docs: https://docs.gaivi.cse.usf.edu/
- Hardware: 15x A100, 6x H100, 10x L40S, plus A40, RTX 6000 Ada, A6000 and older cards.
- Account: request form at https://cseit-usf.atlassian.net/servicedesk/customer/portal/2/create/43 with your NetID.
- Students must name a supervisor who approves the request. Ask the course instructor to be the supervisor. Questions go to cse-tech@usf.edu.
- Jobs on the general partition can be preempted (SIGTERM, then SIGKILL after 5 minutes). Save results after every run. There is a small nopreempt partition.
- JupyterHub: https://jupyterhub.gaivi.cse.usf.edu/ (needs USF VPN).
- Do not run heavy jobs on the login node. Use sbatch.

### CIRCE (USF Research Computing)
- Docs: https://wiki.rc.usf.edu/index.php/CIRCE
- Needs a faculty sponsor. Email rc-help@usf.edu.
- GPUs: V100, L40S (helios partition, ask rc-help), and one GH200. No A100/H100 listed.
- Some partitions have been down in 2026 because of data center cooling issues, so treat it as a backup.

### SC student cluster
- For classes. The instructor emails rc-help@usf.edu with the course ID and roster.
- The hardware page lists no GPUs, so it is probably not useful here.

## Free, no approval

### Kaggle Notebooks - best fallback
- https://www.kaggle.com/docs/notebooks
- 1x P100 16 GB or 2x T4 16 GB, 29 GB RAM, 12 h per session.
- Weekly GPU quota is about 30 h (shown in account settings).
- 29 GB RAM is enough to load a 7B fp16 model on CPU before moving layers to GPU.

### Google Colab free
- Usually one T4 (about 15 GB), 12 h max, limits not published.
- The free Colab Pro offer for US students is closed to new signups.

### Modal
- https://modal.com/pricing
- $30 free credit per month. That is about 14 h on an A100 40GB or about 50 h on a T4.
- Academic grants exist (https://modal.com/academics).

## Academic programs (slower)
- NAIRR Pilot start-up project: up to 2,000 GPU hours, 3-month term, decision in 2 to 3 weeks. Students need a faculty advisor. https://nairrpilot.org/opportunities/startup-project
- NSF ACCESS Explore: undergrads cannot be PI, an advisor or instructor must apply. https://allocations.access-ci.org/project-types
- Google Cloud teaching credits: the instructor applies ($50 per student). https://cloud.google.com/edu/faculty
- Azure for Students ($100) and AWS credits are not useful here. Azure student tiers block GPU VMs and AWS review takes 90 to 120 days.

## Paid fallback (prices on 2026-10-16)
- RunPod: RTX 4090 about $0.34 to $0.89/h, A100 80GB about $1.19 to $1.79/h.
- Vast.ai: RTX 3090 about $0.28/h, RTX 4090 about $0.58/h, A100 80GB about $0.87 to $1.11/h.
- The whole project needs roughly 40 to 60 GPU hours, so the worst case is about $15 to $70.

## Plan
1. Ask the instructor today to approve a GAIVI account.
2. Start on Kaggle (2x T4) while waiting.
3. Use Modal free credit for the SparseGPT runs if GAIVI is not ready.
