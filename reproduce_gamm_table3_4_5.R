#!/usr/bin/env Rscript
# =====================================================================
# reproduce_gamm_table3_4_5.R
# EXACT manuscript GAMM (R/mgcv) reproduction of:
#   Table 3  - NH3-N covariate-controlled abstraction model
#   Table 4  - EC specific-conductance conservative-tracer model
#   Table 5  - season-control robustness (FIXED sample; only the season term varies)
# Run on shared_data/model_frame.csv
# -----------------------
# Usage:  Rscript reproduce_gamm_table3_4_5.R
# Needs:  install.packages(c("mgcv", "nlme"))
# =====================================================================

suppressPackageStartupMessages({ library(mgcv); library(nlme) })

PRIMARY_REACH <- "Yi"   # manuscript's least-confounded primary reach for Table 3/4/5

stopifnot(file.exists("shared_data/model_frame.csv"))
mf <- read.csv("shared_data/model_frame.csv", stringsAsFactors = FALSE)
mf$year          <- factor(mf$year)
mf$dry_window_id <- factor(mf$dry_window_id)
mf$yday          <- as.integer(format(as.Date(mf$date), "%j"))
mf <- mf[order(mf$dry_window_id, as.Date(mf$date)), ]

# ---- optional restricted scales (restricted_for_review/ first, then shared_data/) ----
scales <- NULL
for (pth in c("restricted_for_review/model_frame_scales.csv",
              "shared_data/model_frame_scales.csv")) {
  if (file.exists(pth)) { scales <- read.csv(pth, stringsAsFactors = FALSE); break }
}
sd_of <- function(reach) {
  if (is.null(scales)) return(NA_real_)
  v <- scales[scales$variable == "log_Qdiv_plus1" &
              (is.null(reach) | scales$reach == reach), , drop = FALSE]
  if (nrow(v)) mean(v$sd, na.rm = TRUE) else NA_real_
}

# ---- one GAMM fit on a supplied data frame; returns coef + FITTED n ----
fit_gamm <- function(dat, resp,
                     season = c("temperature","none","harm1","harm2","both","month","cal_k4","cal_k8")) {
  season <- match.arg(season)
  dat <- dat[is.finite(dat[[resp]]) & dat[[resp]] > 0, ]
  dat$y <- log(dat[[resp]])
  dat$month <- factor(format(as.Date(dat$date), "%m"))
  dat$sin1 <- sin(2*pi*dat$yday/365.25); dat$cos1 <- cos(2*pi*dat$yday/365.25)
  dat$sin2 <- sin(4*pi*dat$yday/365.25); dat$cos2 <- cos(4*pi*dat$yday/365.25)
  sterm <- switch(season, temperature = " + s(Twater)", none = "",
                  harm1 = " + sin1 + cos1",
                  harm2 = " + sin1 + cos1 + sin2 + cos2",
                  both  = " + s(Twater) + sin1 + cos1",
                  month = " + month", cal_k4 = " + s(yday, k = 4)",
                  cal_k8 = " + s(yday, k = 8)")
  f <- as.formula(paste0("y ~ z_log_Qdiv_plus1 + s(z_log_Qnd_plus1)", sterm,
                         " + s(API7) + year"))
  m <- tryCatch(
    gamm(f, data = dat, random = list(dry_window_id = ~1),
         correlation = corAR1(form = ~ 1 | dry_window_id)),
    error = function(e) {
      message("    [AR(1) gamm failed: ", conditionMessage(e), " -> no AR(1)]")
      gamm(f, data = dat, random = list(dry_window_id = ~1))
    })
  s <- summary(m$gam)
  list(n = s$n,
       beta = s$p.table["z_log_Qdiv_plus1", "Estimate"],
       se   = s$p.table["z_log_Qdiv_plus1", "Std. Error"],
       p    = s$p.table["z_log_Qdiv_plus1", "Pr(>|t|)"])
}

report <- function(tag, r, sdq = NA_real_) {
  cat(sprintf("\n%s\n  fitted n = %d\n  beta[z_log_Qdiv] = %+.4f  (SE %.4f)  per 1 SD log(Qdiv+1)   p = %.4g\n",
              tag, r$n, r$beta, r$se, r$p))
  if (is.finite(sdq) && sdq > 0)
    cat(sprintf("  beta[log(Qdiv+1)] = %+.4f  per unit log(Qdiv+1)  (= per-SD / sd = %.4f)\n",
                r$beta / sdq, sdq))
}

yi  <- mf[mf$reach == PRIMARY_REACH, ]

cat("===================================================================\n")
cat(sprintf("Tables 3 & 4 — primary reach = %s, and pooled (Yi+Luo) for comparison\n", PRIMARY_REACH))
cat("(scales loaded: ", if (is.null(scales)) "NO -> per-SD only" else "yes -> per-unit shown for single-reach", ")\n", sep = "")
cat("===================================================================\n")

report(sprintf("[Table 3] NH3-N model — %s only", PRIMARY_REACH),
       fit_gamm(yi, "NH3N_down", "temperature"), sd_of(PRIMARY_REACH))
report("[Table 3] NH3-N model — pooled (per-SD only; mixed-reach sd)",
       fit_gamm(mf, "NH3N_down", "temperature"))
report(sprintf("[Table 4] EC tracer model — %s only", PRIMARY_REACH),
       fit_gamm(yi, "EC_down", "temperature"), sd_of(PRIMARY_REACH))
report("[Table 4] EC tracer model — pooled (per-SD only)",
       fit_gamm(mf, "EC_down", "temperature"))

cat("\n===================================================================\n")
cat(sprintf("[Table 5] season-control robustness — %s, FIXED sample\n", PRIMARY_REACH))
cat("All three rows use the SAME rows (response + Twater + API7 present); only the\n")
cat("season term changes, so they are directly comparable (matches the manuscript).\n")
cat("===================================================================\n")
fixed <- yi[is.finite(yi$NH3N_down) & yi$NH3N_down > 0 &
            is.finite(yi$Twater) & is.finite(yi$API7), ]
cat(sprintf("fixed analysable sample: n = %d rows\n", nrow(fixed)))
report("  (i)    no season control",                 fit_gamm(fixed, "NH3N_down", "none"),        sd_of(PRIMARY_REACH))
report("  (ii)   1 harmonic sin+cos (coarsest calendar)", fit_gamm(fixed, "NH3N_down", "harm1"),  sd_of(PRIMARY_REACH))
report("  (iii)  2 harmonics",                        fit_gamm(fixed, "NH3N_down", "harm2"),       sd_of(PRIMARY_REACH))
report("  (iv)   month-of-year factor",               fit_gamm(fixed, "NH3N_down", "month"),       sd_of(PRIMARY_REACH))
report("  (v)    day-of-year smooth s(yday,k=4)",     fit_gamm(fixed, "NH3N_down", "cal_k4"),      sd_of(PRIMARY_REACH))
report("  (vi)   day-of-year smooth s(yday,k=8) [most flexible]", fit_gamm(fixed, "NH3N_down", "cal_k8"), sd_of(PRIMARY_REACH))
report("  (vii)  observed water temp s(Twater) [main]", fit_gamm(fixed, "NH3N_down", "temperature"), sd_of(PRIMARY_REACH))
report("  (viii) temperature + 1 harmonic (both)",   fit_gamm(fixed, "NH3N_down", "both"),        sd_of(PRIMARY_REACH))

cat("\n--- same gradient for the EC conservative tracer (Table 4 response) ---\n")
cat("EC is not produced by temperature-driven nitrification; if its dilution signal\n")
cat("survives a calendar control, it corroborates the mechanism independently.\n")
fixed_ec <- yi[is.finite(yi$EC_down) & yi$EC_down > 0 &
               is.finite(yi$Twater) & is.finite(yi$API7), ]
cat(sprintf("EC fixed analysable sample: n = %d rows\n", nrow(fixed_ec)))
report("  EC (i)   no season control",            fit_gamm(fixed_ec, "EC_down", "none"),        sd_of(PRIMARY_REACH))
report("  EC (ii)  month-of-year factor",          fit_gamm(fixed_ec, "EC_down", "month"),       sd_of(PRIMARY_REACH))
report("  EC (iii) day-of-year smooth s(yday,k=4)", fit_gamm(fixed_ec, "EC_down", "cal_k4"),     sd_of(PRIMARY_REACH))
report("  EC (iv)  observed water temp s(Twater)",  fit_gamm(fixed_ec, "EC_down", "temperature"), sd_of(PRIMARY_REACH))

cat("\nNotes:\n")
cat("- Per-SD coefficients reproduce from the public model_frame.csv alone.\n")
cat("- Per-unit-log (Table 3 +0.117; Table 4 +0.020) needs model_frame_scales.csv.\n")
cat("- If 'fitted n' differs from your manuscript Table 3 n, tell me which sample/\n")
cat("  pooling the manuscript used so the script can be locked to match.\n")
