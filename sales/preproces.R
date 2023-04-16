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



sales$days <- weekdays(as.Date(sales$date))
sales$weeks <- ifelse(sales$days == "Saturday" | sales$days == "Sunday", "Weekend", "Weekday")
sales$month <- as.numeric(format(as.Date(sales$date), "%m"))
sales$year <- as.numeric(format(as.Date(sales$date), "%Y"))
sales$weeknum <- as.POSIXlt(sales$date)
sales$weeknum <- strftime(sales$weeknum,format="%W")

sales$days <- as.factor(sales$days)
sales$weeks <- as.factor(sales$weeks)
sales$weeknum <- as.factor(sales$weeknum)
sales$month <- as.factor(sales$month)
sales$year <- as.factor(sales$year)

sales$family <- as.factor(sales$family)
sales$store_nbr <- as.factor(sales$store_nbr)

lags <- list()
for (lag_offset in 1:30) {
  lags[[paste("lag", lag_offset, sep = "")]] <- (lag(sales$sales, lag_offset))
}

lagged <- sales |>
  mutate(
    !!!lags
  )
lagged

library(gbm)
lagged <- lagged |>
  as_tibble() |>
  select(-id, -date)

fit <- gbm(sales ~ ., data = lagged, n.trees = 1000)


