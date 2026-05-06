# ============================================================================
# TVP-VAR-DY 溢出网络模型
# 导出全部时变切片网络，用于滚动窗口因子模型
# 数据频率: 周度 (由 prepare_data.py 生成的周度收益率/波动率)
# ============================================================================

# 加载必要的包
required_packages <- c("readr", "xts", "zoo")
for(pkg in required_packages){
  if(!require(pkg, character.only = TRUE)){
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# 安装 ConnectednessApproach
if(!require("ConnectednessApproach")){
  library(devtools)
  install_github("GabauerDavid/ConnectednessApproach")
  library(ConnectednessApproach)
}

# -----------------------------------------------------------------------------
# 1. 加载数据
# -----------------------------------------------------------------------------
load_data <- function(data_type = "returns", data_dir = "D:/桌面数据/工作论文/带耦合多层网络/src/data/processed") {
  file_path <- file.path(data_dir,
                         ifelse(data_type == "returns", "industry_returns.csv", "industry_volatility.csv"))
  data <- read.csv(file_path, stringsAsFactors = FALSE)
  data$date <- as.Date(data$date)
  data_xts <- xts(data[, -1], order.by = data$date)

  # 加载行业名称
  names_file <- file.path(data_dir, "industry_names.txt")
  if(file.exists(names_file)){
    industries <- readLines(names_file, encoding = "UTF-8")
    industries <- gsub("^\ufeff", "", industries)
    colnames(data_xts) <- industries
  }

  cat(sprintf("Loaded %s: %d obs x %d industries\n", data_type, nrow(data_xts), ncol(data_xts)))
  return(na.omit(data_xts))
}

# -----------------------------------------------------------------------------
# 2. 行标准化
# -----------------------------------------------------------------------------
row_normalize <- function(W) {
  W_norm <- W
  diag(W_norm) <- 0
  row_sums <- rowSums(W_norm)
  row_sums[row_sums == 0] <- 1
  return(W_norm / row_sums)
}

# -----------------------------------------------------------------------------
# 3. 主函数：导出全部时变邻接矩阵
# -----------------------------------------------------------------------------
run_dy <- function(data_type = "returns",
                   output_dir = "D:/桌面数据/工作论文/带耦合多层网络/src/data/processed") {

  cat(paste(rep("=", 60), collapse=""), "\n")
  cat(sprintf("DY Analysis: %s\n", data_type))
  cat(paste(rep("=", 60), collapse=""), "\n")

  # 加载数据
  data <- load_data(data_type, output_dir)

  # 运行TVP-VAR (周度数据: nlag=1, nfore=4 ≈ 1个月预测步长)
  cat("\nEstimating TVP-VAR (weekly data)...\n")
  dca <- ConnectednessApproach(data,
                               nlag = 1,
                               nfore = 4,
                               model = "TVP-VAR",
                               connectedness = "Time",
                               VAR_config = list(TVPVAR = list(kappa1 = 0.99, kappa2 = 0.96, prior = "BayesPrior")))

  CT <- dca$CT
  K  <- dim(CT)[1]
  T_net <- dim(CT)[3]

  # 日期映射：CT的时间维度对应原始数据的最后 T_net 个日期
  all_dates <- index(data)
  offset <- length(all_dates) - T_net
  net_dates <- all_dates[(offset + 1):length(all_dates)]

  cat(sprintf("Network: %d x %d x %d, dates: %s to %s (offset=%d)\n",
              K, K, T_net, net_dates[1], net_dates[T_net], offset))

  # 堆叠所有邻接矩阵：(K * T_net) x K
  suffix <- ifelse(data_type == "returns", "ret", "vol")
  cat(sprintf("Stacking & normalizing %d adjacency matrices...\n", T_net))

  stacked <- matrix(0, nrow = K * T_net, ncol = K)
  for (t in 1:T_net) {
    adj <- CT[,,t]
    diag(adj) <- 0
    stacked[((t-1)*K + 1):(t*K), ] <- row_normalize(adj)
  }

  # 保存堆叠矩阵
  stacked_file <- file.path(output_dir, sprintf("dy_all_%s.csv", suffix))
  write.table(stacked, stacked_file, sep = ",", row.names = FALSE, col.names = FALSE)

  # 保存日期映射
  dates_df <- data.frame(t_index = 1:T_net, date = as.character(net_dates))
  dates_file <- file.path(output_dir, sprintf("dy_dates_%s.csv", suffix))
  write.csv(dates_df, dates_file, row.names = FALSE)

  cat(sprintf("Saved: %s (%d rows x %d cols)\n", basename(stacked_file), K * T_net, K))
  cat(sprintf("Saved: %s (%d dates)\n", basename(dates_file), T_net))
  cat("Done!\n\n")

  return(list(dca = dca, T_net = T_net, offset = offset))
}

# -----------------------------------------------------------------------------
# 4. 运行
# -----------------------------------------------------------------------------
result_ret <- run_dy("returns")           # 收益率网络
result_vol <- run_dy("volatility")        # 波动率网络
