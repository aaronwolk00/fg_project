suppressPackageStartupMessages({
  library(dplyr); library(stringr); library(lubridate); library(nflreadr)
})
options(nflreadr.verbose = FALSE)

# Optional if you want a truly fresh pull:
# nflreadr::.clear_cache()

dc <- nflreadr::load_depth_charts()

dc <- dc %>%
  mutate(
    dt_parsed = suppressWarnings(lubridate::ymd_hms(dt, tz = "UTC")),
    dt_parsed = if_else(is.na(dt_parsed), as.POSIXct(dt, tz="UTC"), dt_parsed),
    cal_year  = year(dt_parsed),
    cal_month = month(dt_parsed),
    season    = if_else(cal_month <= 2L, cal_year - 1L, cal_year)
  )

# Strict kicker predicate: PK/K only; must be Special Teams; exclude returner/kickoff
dc_pk <- dc %>%
  filter(
    pos_grp == "Special Teams",
    toupper(pos_abb) %in% c("PK","K") |
      str_detect(str_to_lower(pos_name), "^\\s*(place\\s*)?kicker\\s*$")
  ) %>%
  filter(!str_detect(str_to_lower(pos_name), "returner|kickoff"))

message("\n== Unique pos_abb among kicker rows ==")
print(sort(unique(dc_pk$pos_abb)))

message("\n== Sample (ARI, NE, DAL) placekickers only ==")
dc_pk %>%
  filter(team %in% c("ARI","NE","DAL")) %>%
  select(dt, team, player_name, pos_grp, pos_name, pos_abb, pos_rank) %>%
  arrange(team, dt, pos_rank) %>%
  print(n = 200)

message("\n== Any suspicious non-PK/K rows that slipped through? (Should be 0) ==")
dc_pk %>%
  filter(!(toupper(pos_abb) %in% c("PK","K"))) %>%
  count(pos_abb, pos_name, sort = TRUE) %>%
  print(n = 50)
