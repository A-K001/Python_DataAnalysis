import datetime as dt
import pytz # for timezone handling

print("datetime module examples")
print()
print("datetime.datetime class examples")
print()
dnow = dt.datetime.now() #datetime is also a class in datetime module that's why we use dt.datetime / datetime.datetime to access it
print("Current date and time:", dnow) # now() is a class method inside datetime class which is inside datetime module
print("Year:", dnow.year)
print("Month:", dnow.month)
print("Day:", dnow.day)
print("Hour:", dnow.hour)
print()
print("Now custom date and time examples from datetime class")
print()
dcust = dt.datetime(2022, 12, 25, 10, 30, 0) # custom date: year, month, day, hour, minute, second
print("Custom date and time:", dcust)
print("Formatted date:", dcust.strftime("%Y-%m-%d %H:%M:%S")) # formatted date
print("Weekday:", dcust.strftime("%A"))
print("ISO format:", dcust.isoformat())
print("Date only:", dcust.date())
print()
print("Today's date using datetime.date class")
print()
tday = dt.date.today() # date class inside datetime module to get today's date by using today() method in date class
print("Today's date:", tday)
print("Weekday:", tday.strftime("%A"), tday.weekday()) # Monday is 0 and Sunday is 6
print("ISO format:", tday.strftime("%A"), tday.isoweekday()) # Monday is 1 and Sunday is 7
print()
print("Time delta examples using datetime.timedelta class")
print()
tdelta = dt.timedelta(days=10, hours=5, minutes=30) # timedelta class to represent a duration, the difference between two dates or times
future_date = tday + tdelta
print("Future date:", future_date) # adding 10 days, 5 hours and 30 minutes to today's date
past_date = tday - tdelta
print("Past date:", past_date) # subtracting 10 days, 5 hours and 30 minutes from today's date
birthday = dt.date(2004, 1, 14) # my birthday
age = tday - birthday # timedelta = date1 - date2 / date2 = date1 - timedelta
print("Age in years:", age.days // 365) # days since birthday
till_birthday = birthday - tday
print("Days until next birthday:", till_birthday.days if till_birthday.days >= 0 else (dt.date(tday.year + 1, birthday.month, birthday.day) - tday).days)
print()
print("Time class examples using datetime.time class")
print()
t = dt.time(14, 30, 45, 300000 ) # time class to represent time independent of any particular day
print("Time:", t)
print("Hour:", t.hour)
print("Minute:", t.minute)
print("Second:", t.second)
print("Microsecond:", t.microsecond)
print()
print("Combining date and time using datetime.combine() method")
print()
combined = dt.datetime.combine(tday, t) # combine date and time to form a datetime object
print("Combined date and time:", combined)
print()
print("Timezone examples using pytz module")
print()
utc = pytz.utc # UTC timezone
print("UTC timezone:", utc)

eastern = pytz.timezone('US/Eastern') # Eastern timezone
print("Eastern timezone:", eastern)

eastern_time = eastern.localize(dnow) # localize current time to Eastern timezone
print("Current time in Eastern timezone:", eastern_time) 

dt_now = eastern_time.astimezone(pytz.timezone('Asia/Kolkata')) # convert to Asia/Kolkata timezone
print("Current time in Asia/Kolkata timezone:", dt_now)

dt_custom = dt.datetime(2022, 12, 25, 10, 30, 45, tzinfo=pytz.utc) # custom date and time with UTC timezone
print("Custom date and time in UTC timezone:", dt_custom)

dt_now = dt.datetime.now(pytz.timezone('Europe/London')) # current time in London timezone
print("Current time in London timezone:", dt_now)

for tz in pytz.all_timezones[:5]: # print first 5 timezones from all_timezones list
    print(tz)
print()
print("Now examples of strftime and strptime methods") # strftime - datetime to string, strptime - string to datetime
print()
now = dt.datetime.now()
formatted_now = now.strftime("%Y-%m-%d %H:%M:%S") # format current date and time
print("Formatted current date and time:", formatted_now)
parsed_date = dt.datetime.strptime(formatted_now, "%Y-%m-%d %H:%M:%S") # parse formatted date string back to datetime object
print("Parsed date and time:", parsed_date)
print()
print("Converting string to datetime with timezone info")
print()
date_str = "2022-12-25 10:30:45"
dt_naive = dt.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S") # naive datetime (no timezone)
dt_aware = eastern.localize(dt_naive) # make it timezone aware by localizing to Eastern timezone
print("Naive datetime:", dt_naive)
print("Timezone aware datetime:", dt_aware)
print("Converted to UTC timezone:", dt_aware.astimezone(pytz.utc))
print()
print("End of datetime module examples")