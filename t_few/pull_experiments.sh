curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws s3 cp --recursive s3://automl-mm-bench/few_shot/rami_internship/experiments/raft_test_submission_11b experiments/raft_test_submission_11b 
chmod +x experiments/raft_test_submission_11b/run.sh
chmod +x experiments/raft_test_submission_11b/run_p2.sh
chmod +x experiments/raft_test_submission_11b/run_p3.sh
chmod +x experiments/raft_test_submission_11b/run_p4.sh
