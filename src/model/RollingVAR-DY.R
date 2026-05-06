# ============================================================================
# Rolling-Window VAR DY Spillover for Jump-Decomposed Volatility
# Input:  LM jump detection results (10 industries, daily)
# Output: 3 stacked adjacency matrices (CSVt_d, JSVt_zheng_d, JSVt_fu_d)
# ============================================================================

required_packages <- c("readr", "xts", "zoo")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

if (!require("ConnectednessApproach")) {
  library(devtools)
  install_github("GabauerDavid/ConnectednessApproach")
  library(ConnectednessApproach)
}

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PROJECT_DIR <- "D:/桌面数据/工作论文/带耦合多层网络"
LM_DIR      <- file.path(PROJECT_DIR, "data", "raw", "lm_results")
OUTPUT_DIR  <- LM_DIR

CODES <- sprintf("%06d", 32:41)

NLAG        <- 4
NFORE       <- 10
WINDOW_SIZE <- 200
MODEL       <- "VAR"

# -----------------------------------------------------------------------------
# 1. Load LM results: extract one column from 10 CSVs -> xts
# -----------------------------------------------------------------------------
load_lm_data <- function(col_name, data_dir = LM_DIR, codes = CODES) {
  frames <- list()
  for (code in codes) {
    path <- file.path(data_dir, sprintf("%s_lm_har.csv", code))
    df <- read.csv(path, stringsAsFactors = FALSE)
    col_idx <- which(colnames(df) == col_name)
    if (length(col_idx) == 0) stop(sprintf("Column '%s' not found in %s", col_name, path))
    frames[[code]] <- df[, c(1, col_idx)]
    colnames(frames[[code]]) <- c("date", code)
  }

  merged <- frames[[1]]
  for (i in 2:length(codes)) {
    merged <- merge(merged, frames[[codes[i]]], by = "date", all = TRUE)
  }

  merged$date <- as.Date(as.character(merged$date), format = "%Y%m%d")
  merged <- merged[order(merged$date), ]
  data_xts <- xts(merged[, -1], order.by = merged$date)
  # Replace Inf/-Inf with NA, then drop NA rows
  data_xts[!is.finite(data_xts)] <- NA
  data_xts <- na.omit(data_xts)

  cat(sprintf("Loaded '%s': %d obs x %d industries, %s to %s\n",
              col_name, nrow(data_xts), ncol(data_xts),
              index(data_xts)[1], index(data_xts)[nrow(data_xts)]))
  return(data_xts)
}

# -----------------------------------------------------------------------------
# 2. Row normalization (diag -> 0, each row sums to 1)
# -----------------------------------------------------------------------------
row_normalize <- function(W) {
  W_norm <- W
  diag(W_norm) <- 0
  row_sums <- rowSums(W_norm)
  row_sums[row_sums == 0] <- 1
  return(W_norm / row_sums)
}

# -----------------------------------------------------------------------------
# 3. Run rolling-window VAR DY and export stacked adjacency matrices
# -----------------------------------------------------------------------------
run_dy <- function(data_xts, label, output_dir = OUTPUT_DIR) {

  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat(sprintf("Rolling-Window VAR DY: %s\n", label))
  cat(sprintf("  nlag=%d, nfore=%d, window.size=%d\n", NLAG, NFORE, WINDOW_SIZE))
  cat(paste(rep("=", 60), collapse = ""), "\n")

  dca <- ConnectednessApproach(
    data_xts,
    nlag  = NLAG,
    nfore = NFORE,
    window.size = WINDOW_SIZE,
    model = MODEL,
    connectedness = "Time",
    Connectedness_config = list(
      TimeConnectedness = list(generalized = TRUE)
    )
  )

  CT    <- dca$CT
  K     <- dim(CT)[1]
  T_net <- dim(CT)[3]

  all_dates  <- index(data_xts)
  offset     <- length(all_dates) - T_net
  net_dates  <- all_dates[(offset + 1):length(all_dates)]

  cat(sprintf("Network: %d x %d x %d, dates: %s to %s (offset=%d)\n",
              K, K, T_net, net_dates[1], net_dates[T_net], offset))

  stacked <- matrix(0, nrow = K * T_net, ncol = K)
  for (t in 1:T_net) {
    adj <- CT[, , t]
    diag(adj) <- 0
    stacked[((t - 1) * K + 1):(t * K), ] <- adj
  }

  stacked_file <- file.path(output_dir, sprintf("dy_all_%s.csv", label))
  write.table(stacked, stacked_file, sep = ",", row.names = FALSE, col.names = FALSE)

  dates_df   <- data.frame(t_index = 1:T_net, date = as.character(net_dates))
  dates_file <- file.path(output_dir, sprintf("dy_dates_%s.csv", label))
  write.csv(dates_df, dates_file, row.names = FALSE)

  cat(sprintf("Saved: %s (%d rows x %d cols)\n", basename(stacked_file), K * T_net, K))
  cat(sprintf("Saved: %s (%d dates)\n", basename(dates_file), T_net))
  cat("Done!\n\n")

  return(list(dca = dca, T_net = T_net, offset = offset))
}

# -----------------------------------------------------------------------------
# 4. Run for 3 volatility types
# -----------------------------------------------------------------------------
result_csv     <- run_dy(load_lm_data("CSVt_d"),       "csv")
#result_jsv_pos <- run_dy(load_lm_data("JSVt_zheng_d"), "jsv_pos")
#result_jsv_neg <- run_dy(load_lm_data("JSVt_fu_d"),    "jsv_neg")
