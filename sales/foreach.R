library(fpp3)
library(seasonal)
library(ggplot2)
library(gridExtra)
library(dlookr)
library(progressr)
library(future)


sales <- readr::read_csv('D:/contest/kaggle/sales/train.csv')
test <- readr::read_csv('D:/contest/kaggle/sales/test.csv')

# We have to make a forecast for the next half a month
test <- test |>
  as_tsibble(index=date, key=c(store_nbr, family))

# I.
# convert sales to tsibble
sales <- sales |>
  as_tsibble(index = date, key = c(store_nbr, family) )

# number of missing values (explicit missing observations) for each column
sapply(sales, function(x) sum(is.na(x)))

# number of implicit missing observations (gaps)
sales_gaps <- sales |>
  count_gaps(.full = TRUE)

# get a better understanding
# : one day missing from each year.
sales_gaps |>
  group_by(.from) |>
  summarise(count = sum(.n))

sales |>
  filter((month(date) == 12) & (day(date) == 25))

sales <- sales |>
  fill_gaps(sales = median(sales), onpromotion = median(onpromotion), .full = TRUE)

sales <- sales |>
  fill_gaps(.full = TRUE)

# II. ignore (store_nbr, family); yield prediction
# group
sales_grouped <- sales |>
  index_by(date) |>
  summarise(total_sales=sum(sales))
?index_by

# fill_gaps left NA's
sales_grouped |>
  filter((month(date) == 12) & (day(date) == 25))

# fill the NA's with the median
sales_grouped <- sales_grouped |>
  replace_na(list(total_sales=median(sales_grouped$total_sales, na.rm = TRUE)))

# plot moving averages
sales_grouped |>
  mutate(`90-MA` = slider::slide_dbl(total_sales, mean, .before = 89, .after = 0, .complete = TRUE)) |>
  ggplot(aes(x = date)) +
  geom_line(aes(y = total_sales), color = 'grey') +
  geom_line(aes(y = `90-MA`), color = 'orange')




oil <- readr::read_csv('D:/contest/kaggle/sales/oil.csv') |>
  as_tsibble(index = date) |>
  drop_na() |>
  rename(price = dcoilwtico)
holidays <- readr::read_csv('D:/contest/kaggle/sales/holidays_events.csv') |>
  as_tsibble(index = date, key = c(description))

# make oil stationary
oil |>
  ggplot() +
  geom_line(aes(date, price))

oil <- oil |>
  mutate(
    log_price_diff = difference(log(price)),
    price_diff = difference(price)
  ) |>
  drop_na()

p1 <- oil |>
  ggplot() +
  geom_line(aes(x=date, y=price_diff)) 
p2 <- oil |>
  ggplot() +
  geom_line(aes(x=date, y=log_price_diff)) 
grid.arrange(p1,p2, nrow=2)



plan(multisession)
fits <-sales |>
  left_join(oil, by="date") |>
  replace_na(list(
    price_diff = median(oil$price_diff, na.rm = TRUE)
  )) |>
  left_join(holidays, by="date", multiple="any") |>
  replace_na(list(locale = "none", type="none")) |>
  model(
    # ets = ETS(sales),
    # ets_sqrt = ETS(sqrt(sales)),
    # stl_arima_sqrt = decomposition_model(STL(sales), sqrt(sales)),
    arima_regr_price = decomposition_model(
      STL(sales ~ trend() + season(), robust = TRUE),
      ARIMA(season_adjust ~ 1 + type + locale + price_diff + pdq())
    )
  )
plan(sequential)

h <- test |>
  count(day(date)) |>
  nrow()

fcs <- fits |>
  forecast(h=h)

test |>
  left_join(fcs, by=c("store_nbr", "family", "date")) |>
  as_tibble() |>
  select(id, .mean) |>
  rename(sales = .mean) |>
  print() |>
  readr::write_csv("D:/contest/kaggle/sales/submission_ets_foreach.csv")



regr_fcs <- fits |>
  forecast(
    new_data = test |>
      left_join(oil, by="date", multiple = "all") |>
      replace_na(list(
        price_diff = median(oil$price_diff, na.rm = TRUE)
      )) |>
      left_join(holidays, by="date", multiple="all") |>
      replace_na(list(locale = "none", type="none")) |>
      select(-description, -transferred, -locale_name, -log_price_diff, -price, -onpromotion, -id) |>
      select(date, store_nbr, family, price_diff, type, locale) |>
      print()
  ) |>
  filter(.model == "arima_regr_price") |>
  select(-.model)
