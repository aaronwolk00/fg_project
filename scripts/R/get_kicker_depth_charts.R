suppressPackageStartupMessages({
  library(dplyr); library(stringr); library(lubridate)
  library(nflreadr); library(readr); library(jsonlite)
})
options(nflreadr.verbose = FALSE)

TARGET_SEASON <- 2025
OUT_DIR <- file.path("depth_charts", "kickers")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# nflreadr::.clear_cache()  # uncomment once if cache is stale
dc <- nflreadr::load_depth_charts() %>%
  mutate(
    dt_parsed = suppressWarnings(lubridate::ymd_hms(dt, tz = "UTC")),
    dt_parsed = if_else(is.na(dt_parsed), as.POSIXct(dt, tz="UTC"), dt_parsed),
    season    = if_else(month(dt_parsed) <= 2L, year(dt_parsed) - 1L, year(dt_parsed))
  )

base_df <- dc %>% dplyr::filter(season == TARGET_SEASON)

# strict PK filter
pk_all <- base_df %>%
  filter(
    pos_grp == "Special Teams",
    toupper(pos_abb) %in% c("PK","K") |
      str_detect(str_to_lower(pos_name), "^\\s*(place\\s*)?kicker\\s*$")
  ) %>%
  filter(!str_detect(str_to_lower(pos_name), "returner|kickoff")) %>%
  mutate(
    depth_chart_order = coalesce(as.integer(pos_rank), 99L),
    depth_chart       = paste0("K", depth_chart_order)
  )

stopifnot(nrow(pk_all) > 0)

latest_pk_dt <- pk_all %>%
  group_by(team) %>%
  summarise(dt_use = max(dt_parsed, na.rm = TRUE), .groups = "drop")

pk_latest <- pk_all %>%
  inner_join(latest_pk_dt, by = "team") %>%
  filter(dt_parsed == dt_use) %>%
  arrange(team, depth_chart_order, player_name) %>%
  group_by(team, player_name, gsis_id, espn_id) %>%
  slice_min(order_by = depth_chart_order, n = 1, with_ties = FALSE) %>%
  ungroup()

out <- pk_latest %>%
  transmute(
    club_code         = team,
    full_name         = player_name,
    position          = "K",
    depth_position    = "K",
    depth_chart_order = as.integer(depth_chart_order),
    depth_chart       = depth_chart,
    status            = "",
    gsis_id           = gsis_id,
    espn_id           = espn_id,
    pfr_id            = NA_character_,
    player_id         = coalesce(gsis_id, as.character(espn_id)),
    season            = season,
    week              = NA_integer_,
    dt_iso            = format(lubridate::floor_date(dt_parsed, "day"), "%Y-%m-%dT%H:%M:%SZ")
  ) %>%
  distinct(club_code, player_id, .keep_all = TRUE) %>%
  arrange(club_code, depth_chart_order, full_name)

csv_path  <- file.path(OUT_DIR, sprintf("kickers_depth_charts_%d.csv", TARGET_SEASON))
json_path <- file.path(OUT_DIR, sprintf("kickers_depth_charts_%d.json", TARGET_SEASON))

readr::write_csv(out, csv_path)
writeLines(jsonlite::toJSON(out, pretty = TRUE, na = "null", auto_unbox = TRUE), json_path)

cat(sprintf("Wrote:\n  %s  (%d rows, %d teams)\n  %s\n",
            csv_path, nrow(out), dplyr::n_distinct(out$club_code), json_path))

cat("\n== K1 per team (preview) ==\n")
out %>%
  group_by(club_code) %>% slice_min(depth_chart_order, n = 1) %>%
  ungroup() %>% arrange(club_code) %>%
  select(club_code, full_name, depth_chart, dt_iso) %>%
  print(n = 64)
