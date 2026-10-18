suppressPackageStartupMessages({ library(mgcv); library(nlme) })
set.seed(42)
REACH  <- "Yi"
OUT    <- "r2_results"; dir.create(OUT, showWarnings = FALSE)
N_BOOT <- as.integer(Sys.getenv("N_BOOT", "200"))
METHOD   <- Sys.getenv("GAMM_METHOD", "ML")
AR1_FORM <- Sys.getenv("AR1_FORM", "date")
API_TERM <- Sys.getenv("API_TERM", "smooth")
TEMP_K   <- Sys.getenv("TEMP_K", "default")
ENGINE   <- Sys.getenv("ENGINE", "bam")
BAM_METHOD <- Sys.getenv("BAM_METHOD", "fREML")
cat0 <- function(...) cat(sprintf(...), "\n")

f_full <- "restricted_for_review/model_frame_r2_full.csv"
f_pub  <- "shared_data/model_frame_r2.csv"
f_sc   <- "restricted_for_review/model_frame_r2_scales.csv"
if (file.exists(f_full)) {
  mf <- read.csv(f_full, stringsAsFactors = FALSE); zmode <- FALSE
  qn <- c(div = "log_Qdiv_plus1", nd = "log_Qnd_plus1", up = "log_Qup_plus1")
} else {
  stopifnot(file.exists(f_pub))
  mf <- read.csv(f_pub, stringsAsFactors = FALSE); zmode <- TRUE
  qn <- c(div = "z_log_Qdiv_plus1", nd = "z_log_Qnd_plus1", up = "z_log_Qup_plus1")
}
sd_div <- NA_real_
if (zmode && file.exists(f_sc)) {
  sc <- read.csv(f_sc, stringsAsFactors = FALSE)
  v <- sc[sc$reach == REACH & sc$variable == "log_Qdiv_plus1", ]
  if (nrow(v)) sd_div <- v$sd[1]
}
unit_lab <- if (!zmode) "per unit log(Qdiv+1)" else if (is.finite(sd_div)) "per unit log(Qdiv+1) (converted from per-SD)" else "per SD of log(Qdiv+1)"
to_unit <- function(b) if (!zmode) b else if (is.finite(sd_div)) b / sd_div else b

mf <- mf[mf$reach == REACH, ]
mf$date <- as.Date(mf$date)
mf$year <- factor(mf$year); mf$month <- factor(mf$month)
mf$yearmonth <- factor(paste(mf$year, mf$month, sep = "-"))
mf$dry_window_id <- factor(mf$dry_window_id)
for (k in 0:2) {
  suf <- if (k == 0) "" else paste0("_lag", k)
  mf[[paste0("Tdiv", k)]] <- mf[[paste0(qn["div"], suf)]]
  mf[[paste0("Cnd", k)]]  <- mf[[paste0(qn["nd"],  suf)]]
  mf[[paste0("Cup", k)]]  <- mf[[paste0(qn["up"],  suf)]]
}
mf$y_nh3 <- ifelse(is.finite(mf$NH3N_down) & mf$NH3N_down > 0, log(mf$NH3N_down), NA)
mf$y_ec  <- ifelse(is.finite(mf$EC_down) & mf$EC_down > 0, log(mf$EC_down), NA)
mf$y_ecr <- ifelse(is.finite(mf$EC_down) & is.finite(mf$EC_up) & mf$EC_down > 0 & mf$EC_up > 0,
                   log(mf$EC_down / mf$EC_up), NA)
mf$log_Cup <- ifelse(is.finite(mf$NH3N_up) & mf$NH3N_up > 0, log(mf$NH3N_up), NA)
if (!("log_PS_plus1" %in% names(mf)) && "PS_NH3_load" %in% names(mf))
  mf$log_PS_plus1 <- log(pmax(mf$PS_NH3_load, 0) + 1)
mf <- mf[order(mf$dry_window_id, mf$date), ]

fx <- mf[is.finite(mf$y_nh3) & is.finite(mf$y_ec) & is.finite(mf$Twater) & is.finite(mf$API7) &
         is.finite(mf$Tdiv0) & is.finite(mf$Cnd0), ]
fx$dry_window_id <- droplevels(fx$dry_window_id); fx$year <- droplevels(fx$year)
fx$month <- droplevels(fx$month); fx$yearmonth <- droplevels(fx$yearmonth)
cat0("Reach %s | irrigation dry days = %d | analysable sample = %d | with Q_up = %d",
     REACH, nrow(mf), nrow(fx), sum(is.finite(fx$Cup0)))
cat0("Coefficients reported %s", unit_lab)
RESP <- c(y_nh3 = "NH3-N", y_ec = "EC (absolute)", y_ecr = "EC down/up ratio")

fit_one <- function(dat, y, season = "temp", control = "nd", lag = 0, window = "random",
                    extra = "", label = "") {
  dat$y    <- dat[[y]]
  dat$Tdiv <- dat[[paste0("Tdiv", lag)]]
  dat$Ctrl <- dat[[paste0(if (control == "nd") "Cnd" else "Cup", lag)]]
  need <- is.finite(dat$y) & is.finite(dat$Tdiv) & is.finite(dat$Ctrl) & is.finite(dat$Twater) & is.finite(dat$API7)
  if (nzchar(extra)) {
    for (v in unique(unlist(regmatches(extra, gregexpr("[A-Za-z_][A-Za-z0-9_.]*", extra)))))
      if (v %in% names(dat) && is.numeric(dat[[v]])) need <- need & is.finite(dat[[v]])
  }
  dat <- dat[need, ]
  dat$dry_window_id <- droplevels(dat$dry_window_id)
  dat$year <- droplevels(dat$year); dat$month <- droplevels(dat$month); dat$yearmonth <- droplevels(dat$yearmonth)
  tw <- if (TEMP_K == "linear") "Twater" else "s(Twater)"
  sterm <- switch(season, none = "", temp = paste0(" + ", tw), doy = " + s(yday, k = 8)",
                  temp_doy = paste0(" + ", tw, " + s(yday, k = 8)"), month = " + month",
                  yearmonth = " + yearmonth", temp_month = paste0(" + ", tw, " + month"))
  yr <- if (season == "yearmonth" || window == "fixed") "" else " + year"
  wt <- if (window == "fixed") " + dry_window_id" else ""
  api <- if (API_TERM == "smooth") " + s(API7)" else " + API7"
  dat$tt <- as.numeric(as.Date(dat$date) - min(as.Date(dat$date)))
  nwarn <- 0
  quiet <- function(expr) withCallingHandlers(expr, warning = function(w) { nwarn <<- nwarn + 1; invokeRestart("muffleWarning") })
  fail <- function() data.frame(label = label, response = RESP[[y]], season = season, control = control,
                                lag = lag, window = window, extra = extra, n = nrow(dat),
                                beta = NA, se = NA, p = NA, AIC = NA, rho = NA, engine = "FAILED",
                                stringsAsFactors = FALSE)
  if (ENGINE == "bam") {
    re <- if (window == "random") " + s(dry_window_id, bs = \"re\")" else ""
    f <- as.formula(paste0("y ~ Tdiv + s(Ctrl)", api, yr, sterm, extra, wt, re))
    dat <- dat[order(dat$dry_window_id, dat$date), ]
    dat$AR_start <- c(TRUE, dat$dry_window_id[-1] != dat$dry_window_id[-nrow(dat)])
    m0 <- quiet(tryCatch(bam(f, data = dat, method = BAM_METHOD, discrete = FALSE), error = function(e) NULL))
    if (is.null(m0)) return(fail())
    r <- residuals(m0)
    ok <- !dat$AR_start
    # AR(1) coefficient from the residuals of a first fit
    rho <- tryCatch(cor(r[ok], r[which(ok) - 1], use = "complete.obs"), error = function(e) NA_real_)
    if (!is.finite(rho)) rho <- 0
    rho <- max(min(rho, 0.99), -0.99)
    m <- quiet(tryCatch(bam(f, data = dat, method = BAM_METHOD, rho = rho, AR.start = dat$AR_start,
                            discrete = FALSE), error = function(e) NULL))
    if (is.null(m)) { m <- m0; rho <- 0 }
    pt <- summary(m)$p.table
    if (!("Tdiv" %in% rownames(pt))) return(fail())
    eng <- paste0("bam(", BAM_METHOD, ",AR1)")
    if (nwarn > 0) eng <- paste0(eng, " +", nwarn, "w")
    return(data.frame(label = label, response = RESP[[y]], season = season, control = control, lag = lag,
                      window = window, extra = extra, n = nrow(dat),
                      beta = to_unit(pt["Tdiv", "Estimate"]), se = to_unit(pt["Tdiv", "Std. Error"]),
                      p = pt["Tdiv", "Pr(>|t|)"], AIC = AIC(m), rho = rho, engine = eng,
                      stringsAsFactors = FALSE))
  }
  f <- as.formula(paste0("y ~ Tdiv + s(Ctrl)", api, yr, sterm, extra, wt))
  ar1 <- if (AR1_FORM == "date") corAR1(form = ~ tt | dry_window_id) else corAR1(form = ~ 1 | dry_window_id)
  m <- quiet(tryCatch({
    if (window == "random") gamm(f, data = dat, random = list(dry_window_id = ~1), correlation = ar1, method = METHOD)
    else gamm(f, data = dat, correlation = ar1, method = METHOD)
  }, error = function(e) NULL))
  eng <- paste0("gamm+AR1(", AR1_FORM, ",", METHOD, ")")
  if (is.null(m)) {
    m <- quiet(tryCatch(if (window == "random") gamm(f, data = dat, random = list(dry_window_id = ~1), method = METHOD)
                        else gam(f, data = dat, method = METHOD), error = function(e) NULL))
    eng <- "fallback(noAR1)"
  }
  if (nwarn > 0) eng <- paste0(eng, " +", nwarn, "w")
  if (is.null(m)) return(fail())
  g  <- if (inherits(m, "gam")) m else m$gam
  pt <- summary(g)$p.table
  rho <- NA_real_
  if (!inherits(m, "gam")) {
    cs <- m$lme$modelStruct$corStruct
    if (!is.null(cs)) rho <- tryCatch(as.numeric(coef(cs, unconstrained = FALSE))[1], error = function(e) NA)
  }
  data.frame(label = label, response = RESP[[y]], season = season, control = control, lag = lag,
             window = window, extra = extra, n = nrow(dat),
             beta = to_unit(pt["Tdiv", "Estimate"]), se = to_unit(pt["Tdiv", "Std. Error"]),
             p = pt["Tdiv", "Pr(>|t|)"],
             AIC = if (inherits(m, "gam")) AIC(m) else AIC(m$lme), rho = rho, engine = eng,
             stringsAsFactors = FALSE)
}
fmt <- function(r) sprintf("%-42s n=%3d  beta=%+.4f  SE=%.4f  p=%.4f  AIC=%.1f  rho=%.2f  [%s]",
                           r$label, r$n, r$beta, r$se, r$p, r$AIC, r$rho, r$engine)
all_fits <- list()
push <- function(r) { all_fits[[length(all_fits) + 1]] <<- r; cat(fmt(r), "\n"); invisible(r) }

cat0("\nengine = %s | API7 = %s | water temperature = %s", ENGINE, API_TERM, TEMP_K)

cat("\n== Primary specification: s(Twater), Q_nd control, lag 0, random window ==\n")
a_nh3 <- push(fit_one(fx, "y_nh3", "temp", "nd", 0, "random", label = "primary NH3 | temp | Qnd"))
a_ec  <- push(fit_one(fx, "y_ec",  "temp", "nd", 0, "random", label = "primary EC | temp | Qnd"))
a_ecr <- push(fit_one(fx, "y_ecr", "none", "nd", 0, "random", label = "primary EC ratio | none | Qnd"))

cat("\n== Block A: season terms ==\n")
A <- list()
for (y in names(RESP)) for (s in c("none", "temp", "doy", "temp_doy", "month", "yearmonth", "temp_month"))
  A[[length(A) + 1]] <- push(fit_one(fx, y, s, "nd", 0, "random", label = sprintf("A %s | %s", RESP[[y]], s)))
A <- do.call(rbind, A); write.csv(A, file.path(OUT, "r2_blockA_season.csv"), row.names = FALSE)

cat("\n== Block B: dry-window fixed effects ==\n")
wsd <- tapply(fx$Tdiv0, fx$dry_window_id, sd, na.rm = TRUE)
wn  <- tapply(fx$Tdiv0, fx$dry_window_id, function(v) sum(is.finite(v)))
within_var_share <- sum(tapply(fx$Tdiv0, fx$dry_window_id, function(v) sum((v - mean(v))^2))) /
                    sum((fx$Tdiv0 - mean(fx$Tdiv0))^2)
cat0("  windows >= 2 days: %d of %d; with a change in Q_div: %d; within-window share of variance: %.1f%%",
     sum(wn >= 2, na.rm = TRUE), length(wn), sum(wsd > 0, na.rm = TRUE), 100 * within_var_share)
B <- list()
for (y in names(RESP)) for (s in c("none", "temp")) for (lg in 0:1)
  B[[length(B) + 1]] <- push(fit_one(fx, y, s, "nd", lg, "fixed", label = sprintf("B %s | window FE | %s | lag %d", RESP[[y]], s, lg)))
B <- do.call(rbind, B); B$within_var_share <- within_var_share
write.csv(B, file.path(OUT, "r2_blockB_window_fixed.csv"), row.names = FALSE)

cat("\n== Block C: hydrological control x lag ==\n")
C <- list()
for (ctl in c("nd", "up")) {
  cc <- if (ctl == "nd") "Cnd" else "Cup"
  ok <- is.finite(fx$Tdiv0) & is.finite(fx$Tdiv1) & is.finite(fx$Tdiv2) &
        is.finite(fx[[paste0(cc, 0)]]) & is.finite(fx[[paste0(cc, 1)]]) & is.finite(fx[[paste0(cc, 2)]])
  sub <- fx[ok, ]; sub$dry_window_id <- droplevels(sub$dry_window_id)
  for (y in names(RESP)) for (lg in 0:2)
    C[[length(C) + 1]] <- push(fit_one(sub, y, "temp", ctl, lg, "random", label = sprintf("C %s | %s | lag %d", RESP[[y]], ctl, lg)))
}
C <- do.call(rbind, C); write.csv(C, file.path(OUT, "r2_blockC_control_lag.csv"), row.names = FALSE)
for (y in names(RESP)) for (ctl in c("nd", "up")) {
  sub <- C[C$response == RESP[[y]] & C$control == ctl & is.finite(C$AIC), ]
  if (nrow(sub)) { best <- sub[which.min(sub$AIC), ]
    cat0("  %s | %s: lowest AIC at lag %d (beta=%+.4f, p=%.4f, dAIC=%+.1f)",
         RESP[[y]], ctl, best$lag, best$beta, best$p, best$AIC - sub$AIC[sub$lag == 0][1]) }
}

cat("\n== Matched days ==\n")
md <- fx
hi <- md[md$div_class == "high", ]; lo <- md[md$div_class == "low", ]
cat0("  candidate days: high = %d, low = %d", nrow(hi), nrow(lo))

sup <- function(v_hi, v_lo, lab) {
  if (!length(na.omit(v_hi)) || !length(na.omit(v_lo))) return(invisible(NULL))
  qh <- quantile(v_hi, c(.05, .5, .95), na.rm = TRUE); ql <- quantile(v_lo, c(.05, .5, .95), na.rm = TRUE)
  ov <- max(0, min(max(v_hi, na.rm = TRUE), max(v_lo, na.rm = TRUE)) - max(min(v_hi, na.rm = TRUE), min(v_lo, na.rm = TRUE)))
  rg <- max(c(v_hi, v_lo), na.rm = TRUE) - min(c(v_hi, v_lo), na.rm = TRUE)
  cat0("  %-18s high %.1f [%.1f-%.1f]  low %.1f [%.1f-%.1f]  range overlap %.0f%%",
       lab, qh[2], qh[1], qh[3], ql[2], ql[1], ql[3], 100 * ov / max(rg, 1e-9))
}
sup(hi$yday, lo$yday, "day of year"); sup(hi$Twater, lo$Twater, "water temperature")
sup(hi$Cup0, lo$Cup0, "log upstream rel."); sup(hi$Cnd0, lo$Cnd0, "log Q_nd")
cat0("  years: high = {%s}; low = {%s}",
     paste(sort(unique(as.character(hi$year))), collapse = ","), paste(sort(unique(as.character(lo$year))), collapse = ","))

qscale <- if (!zmode) 1 else if (is.finite(sd_div)) 1 / sd_div else 1
LADDER <- list(c(15, 1.5, 0.25), c(21, 2.0, 0.40), c(30, 2.5, 0.60), c(45, 3.0, 1.00))
pairs <- NULL; used_tol <- NULL
for (tol in LADDER) {
  dy_t <- tol[1]; dT_t <- tol[2]; dQ_t <- tol[3] * qscale
  if (!nrow(hi) || !nrow(lo)) break
  cand <- expand.grid(i = seq_len(nrow(lo)), j = seq_len(nrow(hi)))
  cand$dy <- abs(lo$yday[cand$i] - hi$yday[cand$j]); cand$dy <- pmin(cand$dy, 365 - cand$dy)
  cand$dT <- abs(lo$Twater[cand$i] - hi$Twater[cand$j])
  both_up <- is.finite(lo$Cup0[cand$i]) & is.finite(hi$Cup0[cand$j])
  cand$dQ <- ifelse(both_up, abs(lo$Cup0[cand$i] - hi$Cup0[cand$j]), abs(lo$Cnd0[cand$i] - hi$Cnd0[cand$j]))
  cand <- cand[is.finite(cand$dy) & is.finite(cand$dT) & is.finite(cand$dQ) &
               cand$dy <= dy_t & cand$dT <= dT_t & cand$dQ <= dQ_t, ]
  if (!nrow(cand)) { cat0("  tolerance (%d d, %.1f C, %.2f log): 0 admissible pairs", dy_t, dT_t, tol[3]); next }
  cand$dist <- sqrt((cand$dy / dy_t)^2 + (cand$dT / dT_t)^2 + (cand$dQ / dQ_t)^2)
  cand <- cand[order(cand$dist), ]
  ui <- uj <- integer(0); keep <- logical(nrow(cand))
  for (r in seq_len(nrow(cand))) if (!(cand$i[r] %in% ui) && !(cand$j[r] %in% uj)) {
    keep[r] <- TRUE; ui <- c(ui, cand$i[r]); uj <- c(uj, cand$j[r]) }
  pr <- cand[keep, ]
  cat0("  tolerance (%d d, %.1f C, %.2f log): %d 1:1 pairs", dy_t, dT_t, tol[3], nrow(pr))
  if (nrow(pr) >= 5) { pairs <- pr; used_tol <- tol; break }
}
if (!is.null(pairs)) {
  cat0("  tolerance (%d d, %.1f C, %.2f log), %d pairs",
       used_tol[1], used_tol[2], used_tol[3], nrow(pairs))
  res <- lapply(names(RESP), function(y) {
    dlt <- hi[[y]][pairs$j] - lo[[y]][pairs$i]; dlt <- dlt[is.finite(dlt)]
    w <- tryCatch(wilcox.test(dlt, mu = 0, exact = FALSE)$p.value, error = function(e) NA)
    data.frame(response = RESP[[y]], n_pairs = length(dlt), median_diff_log = median(dlt),
               median_pct = 100 * (exp(median(dlt)) - 1), share_positive = mean(dlt > 0), wilcoxon_p = w)
  })
  res <- do.call(rbind, res)
  res$tol_days <- used_tol[1]; res$tol_degC <- used_tol[2]; res$tol_logQ <- used_tol[3]
  print(res, row.names = FALSE)
  write.csv(res, file.path(OUT, "r2_blockC_matched_days.csv"), row.names = FALSE)
  write.csv(data.frame(low_date = lo$date[pairs$i], high_date = hi$date[pairs$j],
                       dy = pairs$dy, dT = pairs$dT, dQ = pairs$dQ),
            file.path(OUT, "r2_blockC_matched_pairs_list.csv"), row.names = FALSE)
} else {
  cat("  fewer than 5 pairs at every tolerance\n")
}

cat("\n== Block D: treatment variation explained by season terms ==\n")
g_doy <- gam(Tdiv0 ~ s(yday, k = 8), data = fx, method = "REML")
g_tw  <- gam(Tdiv0 ~ s(Twater), data = fx, method = "REML")
g_mo  <- tryCatch(lm(Tdiv0 ~ month + year, data = fx), error = function(e) lm(Tdiv0 ~ month, data = fx))
g_ym  <- tryCatch(lm(Tdiv0 ~ yearmonth, data = fx), error = function(e) lm(Tdiv0 ~ month, data = fx))
sd_tot <- sd(fx$Tdiv0)
D <- data.frame(control = c("s(doy,k=8)", "s(Twater)", "month + year FE", "year x month FE"),
                dev_expl_pct = 100 * c(summary(g_doy)$dev.expl, summary(g_tw)$dev.expl, summary(g_mo)$r.squared, summary(g_ym)$r.squared),
                resid_sd = c(sd(residuals(g_doy)), sd(residuals(g_tw)), sd(residuals(g_mo)), sd(residuals(g_ym))))
D$identifying_variation_retained_pct <- 100 * D$resid_sd / sd_tot
print(D, row.names = FALSE)
write.csv(D, file.path(OUT, "r2_blockD_overlap.csv"), row.names = FALSE)

if (N_BOOT > 0) {
  cat0("  dry-window cluster bootstrap, B = %d", N_BOOT)
  wins <- levels(fx$dry_window_id); nb <- 0; out <- matrix(NA, N_BOOT, 3)
  # resample dry windows with replacement
  for (b in seq_len(N_BOOT)) {
    pick <- sample(wins, length(wins), replace = TRUE)
    bs <- do.call(rbind, lapply(seq_along(pick), function(k) { w <- fx[fx$dry_window_id == pick[k], ]; w$dry_window_id <- paste0(pick[k], "_", k); w }))
    bs$dry_window_id <- factor(bs$dry_window_id)
    bt <- fit_one(bs, "y_nh3", "temp", "nd", 0, "random", label = "boot")
    bd <- fit_one(bs, "y_nh3", "doy",  "nd", 0, "random", label = "boot")
    out[b, ] <- c(bt$beta, bd$beta, bt$beta - bd$beta)
    if (b %% 25 == 0) cat0("    %d / %d", b, N_BOOT)
  }
  ok <- is.finite(out[, 3])
  q <- apply(out[ok, , drop = FALSE], 2, quantile, probs = c(0.025, 0.5, 0.975), na.rm = TRUE)
  boot <- data.frame(quantity = c("beta_temp", "beta_doy", "difference"),
                     q025 = q[1, ], median = q[2, ], q975 = q[3, ], share_positive = colMeans(out[ok, , drop = FALSE] > 0), B = sum(ok))
  print(boot, row.names = FALSE)
  write.csv(boot, file.path(OUT, "r2_blockD_bootstrap.csv"), row.names = FALSE)
} else cat("  bootstrap skipped (N_BOOT = 0)\n")

cat("\n== Block E: + s(log C_up) ==\n")
E <- push(fit_one(fx, "y_nh3", "temp", "nd", 0, "random", extra = " + s(log_Cup)", label = "E NH3 | temp | Qnd | + s(log Cup)"))
write.csv(E, file.path(OUT, "r2_blockE_upstreamC.csv"), row.names = FALSE)

cat("\n== Block F: NH3-N fixed-value days removed ==\n")
if ("nh3_clean" %in% names(fx)) {
  n_drop <- sum(fx$nh3_clean == 0)
  cat0("  flagged days: %d of %d (%.0f%%)", n_drop, nrow(fx), 100 * n_drop / nrow(fx))
  fc <- fx[fx$nh3_clean == 1, ]
  fc$dry_window_id <- droplevels(fc$dry_window_id); fc$year <- droplevels(fc$year)
  fc$month <- droplevels(fc$month); fc$yearmonth <- droplevels(fc$yearmonth)
  if (nrow(fc) >= 40) {
    Fb <- list()
    for (y in names(RESP)) for (sn in c("none", "temp", "doy", "month"))
      Fb[[length(Fb) + 1]] <- push(fit_one(fc, y, sn, "nd", 0, "random",
                                           label = sprintf("F %s | clean | %s", RESP[[y]], sn)))
    Fb <- do.call(rbind, Fb); Fb$n_dropped <- n_drop
    write.csv(Fb, file.path(OUT, "r2_blockF_censoring.csv"), row.names = FALSE)
  } else cat0("  only %d days left, not refitted", nrow(fc))
} else {
  cat("  nh3_clean not in the model frame\n")
}

cat("\n== Block G: Table 3 models ==\n")
G <- list()
G[[1]] <- push(fit_one(fx, "y_nh3", "temp", "nd", 0, "random",
                       label = "M1a  primary"))
if (all(c("DO", "turbidity") %in% names(fx))) {
  G[[length(G) + 1]] <- push(fit_one(fx, "y_nh3", "temp", "nd", 0, "random",
                                     extra = " + s(DO) + turbidity",
                                     label = "M2   over-adjustment (+DO +turbidity)"))
} else {
  cat("  DO / turbidity not in the model frame, M2 skipped\n")
}
G[[length(G) + 1]] <- push(fit_one(fx, "y_nh3", "temp", "nd", 0, "random", extra = " + s(log_Cup)",
                                   label = "M1c  + s(log Cup)"))
G <- do.call(rbind, G)
write.csv(G, file.path(OUT, "r2_blockG_table3_rows.csv"), row.names = FALSE)

allf <- do.call(rbind, all_fits)
write.csv(allf, file.path(OUT, "r2_all_fits.csv"), row.names = FALSE)
nh3 <- allf[allf$response == "NH3-N" & is.finite(allf$p), ]
cat("\n== Summary ==\n")
cat0("NH3-N: %d specifications; beta positive in %d; p < 0.05 in %d; beta range %+.3f to %+.3f",
     nrow(nh3), sum(nh3$beta > 0), sum(nh3$p < 0.05), min(nh3$beta), max(nh3$beta))
ec <- allf[allf$response == "EC (absolute)" & is.finite(allf$p), ]
cat0("EC:    %d specifications; beta positive in %d; p < 0.05 in %d; beta range %+.4f to %+.4f",
     nrow(ec), sum(ec$beta > 0), sum(ec$p < 0.05), min(ec$beta), max(ec$beta))
ecr <- allf[allf$response == "EC down/up ratio" & is.finite(allf$p), ]
cat0("EC ratio: %d specifications; beta positive in %d; p < 0.05 in %d", nrow(ecr), sum(ecr$beta > 0), sum(ecr$p < 0.05))
cat0("outputs in ./%s/", OUT)
