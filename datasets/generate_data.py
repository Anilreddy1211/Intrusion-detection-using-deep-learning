import pandas as pd
import numpy as np

# Function to generate normal traffic
def generate_normal_traffic(n):
    data = {
        "duration": np.random.randint(0, 500, n),
        "protocol_type": np.random.choice(["tcp", "udp", "icmp"], n),
        "service": np.random.choice(["http", "ftp", "private", "smtp", "domain_u"], n),
        "flag": np.random.choice(["SF", "S0", "RSTO"], n),
        "src_bytes": np.random.randint(10, 2000, n),
        "dst_bytes": np.random.randint(20, 5000, n),
        "land": np.zeros(n),
        "wrong_fragment": np.zeros(n),
        "urgent": np.zeros(n),
        "hot": np.zeros(n),
        "num_failed_logins": np.zeros(n),
        "logged_in": np.ones(n),
        "num_compromised": np.zeros(n),
        "root_shell": np.zeros(n),
        "su_attempted": np.zeros(n),
        "num_root": np.zeros(n),
        "num_file_creations": np.zeros(n),
        "num_shells": np.zeros(n),
        "num_access_files": np.zeros(n),
        "num_outbound_cmds": np.zeros(n),
        "is_host_login": np.zeros(n),
        "is_guest_login": np.zeros(n),
        "count": np.random.randint(1, 50, n),
        "srv_count": np.random.randint(1, 20, n),
        "serror_rate": np.random.uniform(0, 0.1, n),
        "srv_serror_rate": np.random.uniform(0, 0.1, n),
        "rerror_rate": np.random.uniform(0, 0.1, n),
        "srv_rerror_rate": np.random.uniform(0, 0.1, n),
        "same_srv_rate": np.random.uniform(0.5, 1, n),
        "diff_srv_rate": np.random.uniform(0, 0.3, n),
        "srv_diff_host_rate": np.random.uniform(0, 0.2, n),
        "dst_host_count": np.random.randint(50, 255, n),
        "dst_host_srv_count": np.random.randint(10, 100, n),
        "dst_host_same_srv_rate": np.random.uniform(0.5, 1, n),
        "dst_host_diff_srv_rate": np.random.uniform(0, 0.3, n),
        "dst_host_same_src_port_rate": np.random.uniform(0.5, 1, n),
        "dst_host_srv_diff_host_rate": np.random.uniform(0, 0.2, n),
        "dst_host_serror_rate": np.random.uniform(0, 0.1, n),
        "dst_host_srv_serror_rate": np.random.uniform(0, 0.1, n),
        "dst_host_rerror_rate": np.random.uniform(0, 0.1, n),
        "dst_host_srv_rerror_rate": np.random.uniform(0, 0.1, n),
        "class": ["Normal"] * n
    }
    return pd.DataFrame(data)

# Function to generate attack traffic
def generate_attack_traffic(n):
    data = {
        "duration": np.random.randint(0, 10000, n),
        "protocol_type": np.random.choice(["tcp", "udp", "icmp"], n),
        "service": np.random.choice(["telnet", "ftp", "imap", "smtp", "ssh"], n),
        "flag": np.random.choice(["REJ", "S0", "RSTOS0"], n),
        "src_bytes": np.random.randint(5000, 20000, n),
        "dst_bytes": np.random.randint(0, 500, n),
        "land": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "wrong_fragment": np.random.randint(0, 3, n),
        "urgent": np.random.randint(0, 2, n),
        "hot": np.random.randint(0, 5, n),
        "num_failed_logins": np.random.randint(0, 3, n),
        "logged_in": np.random.choice([0, 1], n),
        "num_compromised": np.random.randint(0, 10, n),
        "root_shell": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "su_attempted": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "num_root": np.random.randint(0, 5, n),
        "num_file_creations": np.random.randint(0, 5, n),
        "num_shells": np.random.randint(0, 3, n),
        "num_access_files": np.random.randint(0, 5, n),
        "num_outbound_cmds": np.zeros(n),
        "is_host_login": np.zeros(n),
        "is_guest_login": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "count": np.random.randint(50, 255, n),
        "srv_count": np.random.randint(50, 100, n),
        "serror_rate": np.random.uniform(0.6, 1, n),
        "srv_serror_rate": np.random.uniform(0.6, 1, n),
        "rerror_rate": np.random.uniform(0.4, 1, n),
        "srv_rerror_rate": np.random.uniform(0.4, 1, n),
        "same_srv_rate": np.random.uniform(0, 0.3, n),
        "diff_srv_rate": np.random.uniform(0.4, 1, n),
        "srv_diff_host_rate": np.random.uniform(0.4, 1, n),
        "dst_host_count": np.random.randint(100, 255, n),
        "dst_host_srv_count": np.random.randint(50, 255, n),
        "dst_host_same_srv_rate": np.random.uniform(0, 0.4, n),
        "dst_host_diff_srv_rate": np.random.uniform(0.5, 1, n),
        "dst_host_same_src_port_rate": np.random.uniform(0, 0.3, n),
        "dst_host_srv_diff_host_rate": np.random.uniform(0.4, 1, n),
        "dst_host_serror_rate": np.random.uniform(0.6, 1, n),
        "dst_host_srv_serror_rate": np.random.uniform(0.6, 1, n),
        "dst_host_rerror_rate": np.random.uniform(0.4, 1, n),
        "dst_host_srv_rerror_rate": np.random.uniform(0.4, 1, n),
        "class": ["Attack"] * n
    }
    return pd.DataFrame(data)

# Generate 60% normal and 40% attack traffic
num_records = 100000  # Total dataset size
num_attacks = int(num_records * 0.20)
num_normal = num_records - num_attacks

# Create datasets
train_data = pd.concat([generate_normal_traffic(num_normal), generate_attack_traffic(num_attacks)]).sample(frac=1).reset_index(drop=True)
test_data = pd.concat([generate_normal_traffic(num_normal // 2), generate_attack_traffic(num_attacks // 2)]).sample(frac=1).reset_index(drop=True)

# Save datasets
train_data.to_csv("Train_data_1.csv", index=False)
test_data.to_csv("Test_data_1.csv", index=False)

print("Train and Test datasets with ~40% attack probability generated successfully!")
