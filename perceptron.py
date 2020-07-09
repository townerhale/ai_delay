import pandas
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import classification_report,confusion_matrix



##creates a training and test set for x and y vectors for preprocessing the data
##uses a specific classifier to then classifiy the data
def perceptron(vector_x, vector_y):
    """Creates a training and test set for x and y matrixes for preprocessing data
        Then uses a specific classifier (in this case MLP) to classify the data.
    Args:
        vector_x(obj): Holds a matrix
        vector_y (obj): Holds a y matrix

    Returns:
        Prints both a confusion_matrix and classification report
    """

    X_train, X_test, y_train, y_test = train_test_split(vector_x, vector_y)
    scaler = StandardScaler()  ##standardizes the data for outliers

    scaler.fit(X_train)   ## each column will have a mean of 0 and standard deviation = 1
    X_train = scaler.transform(X_train) ##transforms the tests data so each feature is on the same scale
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(31,31,31), max_iter=500)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)

    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test, predictions))



def determine_peak(time):
    """Determines if it is peak or nonpeak time during the weekend
       Peak time is before 9:30 am and between 16:29-18:34
       Args:
           time(obj): Holds either the pta(arrival time of the station) or ptd(departure time of station)

       Returns:
           returns 1 if peak, return 0 if not_peak
       """

    if time < datetime.strptime('09:30', '%H:%M').time():
        return 1
    elif time > datetime.strptime('16:29', '%H:%M').time():
        if time > datetime.strptime('18:34', '%H:%M').time():
            return 0
        else:
            return 1
    else:
        return 0

def determine_time(time):
    """Determines what time it is from the data so it can be used as one feature for the  matrix
           Args:
               time(obj): Holds either the pta(arrival time of the station) or ptd(departure time of station)

           Returns:
               " ": returns a string that states the time
           """
    if time >= datetime.strptime('06:00', '%H:%M').time() and time < datetime.strptime('07:00', '%H:%M').time() :
        return "six"
    elif time >= datetime.strptime('07:00', '%H:%M').time() and time < datetime.strptime('08:00', '%H:%M').time():
            return "seven"
    elif time >= datetime.strptime('08:00', '%H:%M').time() and time < datetime.strptime('09:00', '%H:%M').time():
        return "eight"
    elif time >= datetime.strptime('09:00', '%H:%M').time() and time < datetime.strptime('10:00', '%H:%M').time():
        return "nine"
    elif time >= datetime.strptime('10:00', '%H:%M').time() and time < datetime.strptime('11:00', '%H:%M').time():
        return "ten"
    elif time >= datetime.strptime('11:00', '%H:%M').time() and time < datetime.strptime('12:00', '%H:%M').time():
        return "eleven"
    elif time >= datetime.strptime('12:00', '%H:%M').time() and time < datetime.strptime('13:00', '%H:%M').time():
        return "twelve"
    elif time >= datetime.strptime('13:00', '%H:%M').time() and time < datetime.strptime('14:00', '%H:%M').time():
        return "one pm"
    elif time >= datetime.strptime('14:00', '%H:%M').time() and time < datetime.strptime('15:00', '%H:%M').time():
        return "two pm"
    elif time >= datetime.strptime('15:00', '%H:%M').time() and time < datetime.strptime('16:00', '%H:%M').time():
        return "three pm"
    elif time >= datetime.strptime('16:00', '%H:%M').time() and time < datetime.strptime('17:00', '%H:%M').time():
        return "four pm"
    elif time >= datetime.strptime('17:00', '%H:%M').time() and time < datetime.strptime('18:00', '%H:%M').time():
        return "five pm"
    elif time >= datetime.strptime('18:00', '%H:%M').time() and time < datetime.strptime('19:00', '%H:%M').time():
        return "six pm"
    elif time >= datetime.strptime('19:00', '%H:%M').time() and time < datetime.strptime('20:00', '%H:%M').time():
        return "seven pm"
    elif time >= datetime.strptime('20:00', '%H:%M').time() and time < datetime.strptime('21:00', '%H:%M').time():
        return "eight pm"
    elif time >= datetime.strptime('21:00', '%H:%M').time() and time < datetime.strptime('22:00', '%H:%M').time():
        return "nine pm"
    elif time >= datetime.strptime('22:00', '%H:%M').time() and time < datetime.strptime('23:00', '%H:%M').time():
        return "ten pm"
    elif time >= datetime.strptime('23:00', '%H:%M').time() and time < datetime.strptime('00:00', '%H:%M').time():
        return "eleven pm"
    elif time >= datetime.strptime('00:00', '%H:%M').time() and time < datetime.strptime('01:00', '%H:%M').time():
        return "twelve am"
    else:
        return "one am"



def find_day(row):
    """Finds which specific day of the week it is and uses it as a feature
      Also accounts for which month it is.
           Args:
               row(obj): contains the row matrix which holds the data for which day it is

           Returns:
               mon, tues, wed, thur, fri, sat, sun: each holding a value of either 1 or 0 indicating what day it is

           """
    count = '2'
    mon, tues, wed, thur, fri, sat, sun = (0, 0, 0, 0, 0, 0, 0)

    if '201803' in row.rid:     #the last number represents the mounth
        count = '3'
    if '201801' in row.rid:
        count = '1'
    if '201804' in row.rid:
        count = '4'
    if '201805' in row.rid:
        count = '5'
    if '201806' in row.rid:
        count = '6'
    if '201807' in row.rid:
        count = '7'

    if ('20180' + str(count)+ '01') in row.rid: #the last number represents the day
        if count == '1':
            mon = 1
        elif count == '4' or count == '6':
            sun = 1
        elif count == '5':
            tues = 1
        elif count == '6':
            fri = 1
        else:
            thur = 1

    if ('20180' + str(count) + '02') in row.rid:
        if count == '1':
            tues = 1
        elif count == '4'or count == '6':
            mon = 1
        elif count == '5':
            wed = 1
        elif count == '6':
            sat = 1
        else:
            fri = 1
    if ('20180' + str(count) + '03') in row.rid:
        if count == '1':
            wed = 1
        elif count == '4' or count == '6':
            tues = 1
        elif count == '5':
            thur = 1
        elif count == '6':
            sun = 1
        else:
            sat = 1
    if ('20180' + str(count) + '04') in row.rid:
        if count == '1':
            thur = 1
        elif count == '4' or count == '6':
            wed = 1
        elif count == '5':
            fri = 1
        elif count == '6':
            mon = 1
        else:
            sun = 1
    if ('20180' + str(count) + '05') in row.rid:
        if count == '1':
            fri = 1
        elif count == '4' or count == '6':
            thur = 1
        elif count == '5':
            sat = 1
        elif count == '6':
            tues = 1
        else:
            mon = 1
    if ('20180' + str(count) + '06') in row.rid:
        if count == '1':
            sat = 1
        elif count == '4' or count == '6':
            fri = 1
        elif count == '5':
            sun = 1
        elif count == '6':
            wed = 1
        else:
            tues = 1
    if ('20180' + str(count) + '07') in row.rid:
        if count == '1':
            sun = 1
        elif count == '4' or count == '6':
            sat = 1
        elif count == '5':
            mon = 1
        elif count == '6':
            thur = 1
        else:
            wed = 1
    if ('20180' + count + '08') in row.rid:
        if count == '1':
            mon = 1
        elif count == '4' or count == '6':
            sun = 1
        elif count == '5':
            tues = 1
        elif count == '6':
            fri = 1
        else:
            thur = 1
    if ('20180' + str(count) + '09') in row.rid:
        if count == '1':
            tues = 1
        elif count == '4' or count == '6':
            mon = 1
        elif count == '5':
            wed = 1
        elif count == '6':
            sat = 1
        else:
            fri = 1
    if ('20180' + str(count) + '10') in row.rid:
        if count == '1':
            wed = 1
        elif count == '4' or count == '6':
            tues = 1
        elif count == '5':
            thur = 1
        elif count == '6':
            sun = 1
        else:
            sat = 1
    if ('20180' + str(count) + '11') in row.rid:
        if count == '1':
            thur = 1
        elif count == '4' or count == '6':
            wed = 1
        elif count == '5':
            fri = 1
        elif count == '6':
            mon = 1
        else:
            sun = 1
    if ('20180' + str(count) + '12') in row.rid:
        if count == '1':
            fri = 1
        elif count == '4' or count == '6':
            thur = 1
        elif count == '5':
            sat = 1
        elif count == '6':
            tues = 1
        else:
            mon = 1
    if ('20180' + str(count) + '13') in row.rid:
        if count == '1':
            sat = 1
        elif count == '4' or count == '6':
            fri = 1
        elif count == '5':
            sun = 1
        elif count == '6':
            wed = 1
        else:
            tues = 1
    if ('20180' + str(count) + '14') in row.rid:
        if count == '1':
            sun = 1
        elif count == '4' or count == '6':
            sat = 1
        elif count == '5':
            mon = 1
        elif count == '6':
            thur = 1
        else:
            wed = 1
    if ('20180' + count + '15') in row.rid:
        if count == '1':
            mon = 1
        elif count == '4' or count == '6':
            sun = 1
        elif count == '5':
            tues = 1
        elif count == '6':
            fri = 1
        else:
            thur = 1
    if ('20180' + str(count) + '16') in row.rid:
        if count == '1':
            tues = 1
        elif count == '4' or count == '6':
            mon = 1
        elif count == '5':
            wed = 1
        elif count == '6':
            sat = 1
        else:
            fri = 1
    if ('20180' + str(count) + '17') in row.rid:
        if count == '1':
            wed = 1
        elif count == '4' or count == '6':
            tues = 1
        elif count == '5':
            thur = 1
        elif count == '6':
            sun = 1
        else:
            sat = 1
    if ('20180' + str(count) + '18') in row.rid:
        if count == '1':
            thur = 1
        elif count == '4' or count == '6':
            wed = 1
        elif count == '5':
            fri = 1
        elif count == '6':
            mon = 1
        else:
            sun = 1
    if ('20180' + str(count) + '19') in row.rid:
        if count == '1':
            fri = 1
        elif count == '4' or count == '6':
            thur = 1
        elif count == '5':
            sat = 1
        elif count == '6':
            tues = 1
        else:
            mon = 1
    if ('20180' + str(count) + '20') in row.rid:
        if count == '1':
            sat = 1
        elif count == '4' or count == '6':
            fri = 1
        elif count == '5':
            sun = 1
        elif count == '6':
            wed = 1
        else:
            tues = 1
    if ('20180' + str(count) + '21') in row.rid:
        if count == '1':
            sun = 1
        elif count == '4' or count == '6':
            sat = 1
        elif count == '5':
            mon = 1
        elif count == '6':
            thur = 1
        else:
            wed = 1
    if ('20180' + count + '22') in row.rid:
        if count == '1':
            mon = 1
        elif count == '4' or count == '6':
            sun = 1
        elif count == '5':
            tues = 1
        elif count == '6':
            fri = 1
        else:
            thur = 1
    if ('20180' + str(count) + '23') in row.rid:
        if count == '1':
            tues = 1
        elif count == '4' or count == '6':
            mon = 1
        elif count == '5':
            wed = 1
        elif count == '6':
            sat = 1
        else:
            fri = 1
    if ('20180' + str(count) + '24') in row.rid:
        if count == '1':
            wed = 1
        elif count == '4' or count == '6':
            tues = 1
        elif count == '5':
            thur = 1
        elif count == '6':
            sun = 1
        else:
            sat = 1
    if ('20180' + str(count) + '25') in row.rid:
        if count == '1':
            thur = 1
        elif count == '4' or count == '6':
            wed = 1
        elif count == '5':
            fri = 1
        elif count == '6':
            mon = 1
        else:
            sun = 1
    if ('20180' + str(count) + '26') in row.rid:
        if count == '1':
            fri = 1
        elif count == '4' or count == '6':
            thur = 1
        elif count == '5':
            sat = 1
        elif count == '6':
            tues = 1
        else:
            mon = 1
    if ('20180' + str(count) + '27') in row.rid:
        if count == '1':
            sat = 1
        elif count == '4' or count == '6':
            fri = 1
        elif count == '5':
            sun = 1
        elif count == '6':
            wed = 1
        else:
            tues = 1
    if ('20180' + str(count) + '28') in row.rid:
        if count == '1':
            sun = 1
        elif count == '4' or count == '6':
            sat = 1
        elif count == '5':
            mon = 1
        elif count == '6':
            thur = 1
        else:
            wed = 1
    return mon, tues, wed, thur, fri, sat, sun

def liverpool_model(df):
    """
          Determines if there is a delay between predicted and arrival time at liverpool station
               Args:
                   df(obj): a DataFrame object that reads the csv file

               Returns:
                   liverpool_delay(obj): a list that contains the delay time

               """

    liverpool_delay = []

    df = df.loc[df['tpl'] == 'LIVST']  ##tpl is the station column

    for row in df.itertuples():  ##go through each row in the liverpool column
        if row.ptd == '\\N' or row.dep_at == '\\N': ##if there isn't any data
            pass
        else:

            ##get if delay, and get exact delay
            start_dt = datetime.strptime(row.ptd, '%H:%M')  #ptd = predicted departure time
            end_dt = datetime.strptime(row.dep_at, '%H:%M') #dep_at = actual departure time

            diff = (end_dt - start_dt)
            diff2 = (start_dt - end_dt)

            if (diff.seconds < diff2.seconds):  ##accounts for 21:00 and 8:00 the next morning so two different days
                final_delay = diff.seconds
            else:
                final_delay = diff2.seconds             ##print("delay " + str(final_delay))


            liverpool_delay.append(final_delay)

    return liverpool_delay

def chlmsfield_model(df, liverpool_delay):
    """
        Finds different features and places them in the x vector,
        put 0 for no arrival delay in y vector because using final Norwich model
        and find the the specific arrival delay amount for the chlmsfield station

            Args:
                df(obj): a DataFrame object that reads the csv file
                liverpool_delay(obj): a list that holds the liverpool delay amount

            Returns:
                vector_x(obj): a matrix that holds different features
                vector_y(obj): a matrix that holds a 1 for delay or 0 for no delay
                chlm_delay(obj): a list that holds the chlmsfield delay amount

                  """
    six, seven, eight, nine, ten, eleven, twelve, one_pm, two_pm, three_pm, four_pm, five_pm, six_pm, seven_pm, eight_pm, nine_pm, ten_pm, eleven_pm, twelve_am, one_am = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) #
    vector_x = []
    vector_y = []
    hold_liverpool_delay = liverpool_delay
    index = 0

    delay_arr = 0
    extra_weight_no_delay = 0 ##add a extra feature to account for if there is no delay

    df = df.loc[df['tpl'] == 'CHLMSFD']


    chlm_delay = []

    for row in df.itertuples():

        """
        The commented out code below was the original code used before moving on to
        to the next model, in which the planned and actual arrival time variables are 
        now changed to planned and actual departure time variables 
        """
        # if not liverpool_delay:  ##if there isn't a delay pass
        #     pass
        # else:
        #     hold_liverpool_delay = liverpool_delay.pop(0)
        # if row.pta == '\\N' or row.arr_at == '\\N':
        #     pass

        if row.ptd == '\\N' or row.dep_at == '\\N':
            pass
        else:
            ##satisfies what day it is digit
            mon, tues, wed, thur, fri, sat, sun = find_day(
                row)  ##print("mon: " + str(mon) + " tues: " + str(tues) + " wed: " + str(wed) + " thurs: " + str(thur) + " fri: " + str(fri))


            ptd = datetime.strptime(row.ptd, '%H:%M').time() ##used to determine peak time which will be referenced later

            ##get if delay, and get exact delay for departure time
            start_dt = datetime.strptime(row.ptd, '%H:%M')
            end_dt = datetime.strptime(row.dep_at, '%H:%M')
            diff = (end_dt - start_dt)
            diff2 = (start_dt - end_dt)
            if (diff.seconds < diff2.seconds):  #convert between days
                final_delay = diff.seconds
            else:
                final_delay = diff2.seconds

            chlm_delay.append(final_delay) #append to list the delay amount

            ##get if delay, and get exact delay for arrival times, used in original model
            # start_dt_arr = datetime.strptime(row.pta, '%H:%M')
            # end_dt_arr = datetime.strptime(row.arr_at, '%H:%M')
            # diff_arr = (end_dt_arr - start_dt_arr)
            # diff2_arr = (start_dt_arr - end_dt_arr)
            # if (diff_arr.seconds < diff2_arr.seconds):  ##convert between days
            #     final_delay_arr = diff_arr.seconds
            # else:
            #     final_delay_arr = diff2_arr.seconds
            #     ##print("delay " + str(final_delay))
            #
            #     ##figure out of there is a delay
            # if final_delay_arr > 0:
            #     delay_arr = 1
            # else:
            #     delay_arr = 0
            #     extra_weight_no_delay = 0

            vector_y.append(0) ##there is no arrival time delay since this code is changed to account for next models

            ## vector_y.append(final_delay_arr) ##used in original model

            ##satisfies what day it is digit, sometimes used as a feature, sometimes not
            mon, tues, wed, thur, fri, sat, sun = find_day(
                row)


            ##satisfies what time it is
            hold_time = determine_time(ptd)
            if hold_time == "six":
                six = 1
            elif hold_time == "seven":
                seven = 1
            elif hold_time == "eight":
                eight = 1
            elif hold_time == "nine":
                nine = 1
            elif hold_time == "ten":
                ten = 1
            elif hold_time == "eleven":
                eleven = 1
            elif hold_time == "twelve":
                twelve = 1
            elif hold_time == "one pm":
                one_pm = 1
            elif hold_time == "two pm":
                two_pm = 1
            elif hold_time == "three pm":
                three_pm = 1
            elif hold_time == "four pm":
                four_pm = 1
            elif hold_time == "five pm":
                five_pm = 1
            elif hold_time == "six pm":
                six_pm = 1
            elif hold_time == "seven pm":
                seven_pm = 1
            elif hold_time == "eight pm":
                eight_pm = 1
            elif hold_time == "nine pm":
                nine_pm = 1
            elif hold_time == "ten pm":
                ten_pm = 1
            elif hold_time == "eleven pm":
                eleven_pm = 1
            elif hold_time == "twelve am":
                twelve_am = 1
            else:
                one_am = 1

            ##satisfies if peak or not peak
            if sat == 1 or sun == 1:
                peak = 0
            else:
                peak = determine_peak(0, ptd)

            """
            In other experiments Vector x added the features  'peak', and 'weekend', and 'monday''tues'
             'wednesday' 'thursday' 'friday' to determine if the inclusion of these days had an impact
            determine if there was an impact,
            """

            vector_x.append(
                [extra_weight_no_delay, hold_liverpool_delay, delay_arr, mon, tues, wed, thur, fri, sat, sun, six, seven,
                 eight, nine, ten, eleven, twelve, one_pm, two_pm, three_pm, four_pm, five_pm, six_pm, seven_pm,
                 eight_pm, nine_pm, ten_pm, eleven_pm, twelve_am, one_am])
            index = 1 + index
    return vector_y, vector_x, chlm_delay

def colchester_model(df, liverpool_delay, chlem_delay):
    """
           Finds different features and places them in the x vector,
           put a 0 for no arrival delay in y vector because it's using the final Norwich model
           and find the the specific arrival delay amount for the colchester station

               Args:
                   df(obj): a DataFrame object that reads the csv file
                   liverpool_delay(obj): a list that holds the liverpool delay amount
                   chlem_delay(obj): a list that holds the chlemsfield elay amount

               Returns:
                   vector_x(obj): a matrix that holds different features
                   vector_y(obj): a matrix that holds a 1 for delay or 0 for no delay
                   colch_delay(obj): a list that holds the chlmsfield delay amount

                     """

    six, seven, eight, nine, ten, eleven, twelve, one_pm, two_pm, three_pm, four_pm, five_pm, six_pm, seven_pm, eight_pm, nine_pm, ten_pm, eleven_pm, twelve_am, one_am = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    vector_x = []
    vector_y = []
    hold_liverpool_delay = liverpool_delay
    hold_clem_delay = chlem_delay
    colch_delay = []
    delay_arr = 0
    extra_weight_no_delay = 0
    df = df.loc[df['tpl'] == 'CLCHSTR']


    for row in df.itertuples():

        # if not chlem_delay and liverpool_delay:
        #     pass
        # else:
        #     hold_liverpool_delay = liverpool_delay.pop(0)
        #     hold_clem_delay = chlem_delay.pop(0)
        # if row.pta == '\\N' or row.arr_at == '\\N':
        #     pass
        if row.ptd == '\\N' or row.dep_at == '\\N':
            pass
        else:
            ##satisfies what day it is digit, sometimes used as a feature, sometimes not
            mon, tues, wed, thur, fri, sat, sun = find_day(
                row)

            ptd = datetime.strptime(row.ptd, '%H:%M').time() ##used to determine peak time which will be used later

            ##get if delay, and get exact delay
            start_dt = datetime.strptime(row.ptd, '%H:%M')
            end_dt = datetime.strptime(row.dep_at, '%H:%M')
            diff = (end_dt - start_dt)
            diff2 = (start_dt - end_dt)
            if (diff.seconds < diff2.seconds):  ##convert between days
                final_delay = diff.seconds
            else:
                final_delay = diff2.seconds

            colch_delay.append(final_delay)


            ##get if delay, and get exact delay for arrival
            # start_dt_arr = datetime.strptime(row.pta, '%H:%M')
            # end_dt_arr = datetime.strptime(row.arr_at, '%H:%M')
            # diff_arr = (end_dt_arr - start_dt_arr)
            # diff2_arr = (start_dt_arr - end_dt_arr)
            # if (diff_arr.seconds < diff2_arr.seconds):  ##convert between days
            #     final_delay_arr = diff_arr.seconds
            # else:
            #     final_delay_arr = diff2_arr.seconds
            #     ##print("delay " + str(final_delay))
            #
            #     ##figure out of there is a delay
            # if final_delay_arr > 0:
            #     delay_arr = 1
            # else:
            #     delay_arr = 0
            #     extra_weight_no_delay = 0

            # vector_y.append(final_delay_arr)

            ##satisfies what time it is
            hold_time = determine_time(ptd)
            if hold_time == "six":
                six = 1
            elif hold_time == "seven":
                seven = 1
            elif hold_time == "eight":
                eight = 1
            elif hold_time == "nine":
                nine = 1
            elif hold_time == "ten":
                ten = 1
            elif hold_time == "eleven":
                eleven = 1
            elif hold_time == "twelve":
                twelve = 1
            elif hold_time == "one pm":
                one_pm = 1
            elif hold_time == "two pm":
                two_pm = 1
            elif hold_time == "three pm":
                three_pm = 1
            elif hold_time == "four pm":
                four_pm = 1
            elif hold_time == "five pm":
                five_pm = 1
            elif hold_time == "six pm":
                six_pm = 1
            elif hold_time == "seven pm":
                seven_pm = 1
            elif hold_time == "eight pm":
                eight_pm = 1
            elif hold_time == "nine pm":
                nine_pm = 1
            elif hold_time == "ten pm":
                ten_pm = 1
            elif hold_time == "eleven pm":
                eleven_pm = 1
            elif hold_time == "twelve am":
                twelve_am = 1
            else:
                one_am = 1

            ##satisfies if peak or not peak
            if sat == 1 or sun == 1:
                peak = 0
            else:
                peak = determine_peak(ptd)

            vector_y.append(0)

            """
                In other experiments Vector x added the features  'peak', and 'weekend', and 'monday''tues'
                'wednesday' 'thursday' 'friday' to determine if the inclusion of these days had an impact
                determine if there was an impact
            """
            vector_x.append(
                [extra_weight_no_delay, hold_liverpool_delay, hold_clem_delay, delay_arr, mon, tues, wed, thur, fri,
                 sat, sun, six, seven, eight, nine, ten, eleven, twelve, one_pm, two_pm, three_pm, four_pm, five_pm,
                 six_pm, seven_pm, eight_pm, nine_pm, ten_pm, eleven_pm, twelve_am, one_am])

    return vector_y, vector_x, colch_delay

def norwich_model(df, liverpool_delay, chlem_delay, colch_delay):
    """
              Finds different features and places them in the x vector,
              put a 0 for no delay in y vector because it's using the final Norwich model
              and find the the specific delay amount for the chlmsfield station

                  Args:
                      df(obj): a DataFrame object that reads the csv file
                      liverpool_delay(obj): a list that holds the liverpool delay amount
                      chlem_delay(obj): a list that holds the chlemsfield delay amount
                      colch_delay(obj): a list that holds the colchester delay amount

                  Returns:
                      vector_x(obj): a matrix that holds different features
                      vector_y(obj): a matrix that holds a 1 for delay or 0 for no delay

                        """
    six, seven, eight, nine, ten, eleven, twelve, one_pm, two_pm, three_pm, four_pm, five_pm, six_pm, seven_pm, eight_pm, nine_pm, ten_pm, eleven_pm, twelve_am, one_am = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    vector_x = []
    vector_y = []

    #hold_liverpool_delay = liverpool_delay #not used in best model
    #hold_clem_delay = chlem_delay
    #hold_colch_delay = colch_delay

    index = 0
    #delay = 0 #holds whether or not there is a delay

    extra_weight_no_delay = 0
    df = df.loc[df['tpl'] == 'CLCHSTR']

    #more_likely_60 = 0 #not used in best model

    for row in df.itertuples():
        mon, tues, wed, thur, fri, sat, sun = (0, 0, 0, 0, 0, 0, 0)
        if not chlem_delay and liverpool_delay and colch_delay:
            pass
        else:
            hold_liverpool_delay = liverpool_delay.pop(0)
            hold_clem_delay = chlem_delay.pop(0)
            hold_colch_delay = colch_delay.pop(0)
            if row.pta == '\\N' or row.arr_at == '\\N':
                pass
            else:
                ##satisfies what day it is digit
                mon, tues, wed, thur, fri, sat, sun = find_day(row)     ##print("mon: " + str(mon) + " tues: " + str(tues) + " wed: " + str(wed) + " thurs: " + str(thur) + " fri: " + str(fri))
                pta = datetime.strptime(row.pta, '%H:%M').time()

                ##get if delay, and get exact delay for arrival
                start_dt_arr = datetime.strptime(row.pta, '%H:%M')
                end_dt_arr = datetime.strptime(row.arr_at, '%H:%M')
                diff_arr = (end_dt_arr - start_dt_arr)
                diff2_arr = (start_dt_arr - end_dt_arr)
                if (diff_arr.seconds < diff2_arr.seconds):  ##convert between days
                    final_delay_arr = diff_arr.seconds
                else:
                    final_delay_arr = diff2_arr.seconds
                    ##print("delay " + str(final_delay))

                    ##figure out of there is a delay
                if final_delay_arr > 0:
                    delay_arr = 1
                else:
                    delay_arr = 0
                    extra_weight_no_delay = 0

                vector_y.append(final_delay_arr)

                ##satisfies what time it is
                hold_time = determine_time(pta)
                if hold_time == "six":
                    six = 1
                elif hold_time == "seven":
                    seven = 1
                elif hold_time == "eight":
                    eight = 1
                elif hold_time == "nine":
                    nine = 1
                elif hold_time == "ten":
                    ten = 1
                elif hold_time == "eleven":
                    eleven = 1
                elif hold_time == "twelve":
                    twelve = 1
                elif hold_time == "one pm":
                    one_pm = 1
                elif hold_time == "two pm":
                    two_pm = 1
                elif hold_time == "three pm":
                    three_pm = 1
                elif hold_time == "four pm":
                    four_pm = 1
                elif hold_time == "five pm":
                    five_pm = 1
                elif hold_time == "six pm":
                    six_pm = 1
                elif hold_time == "seven pm":
                    seven_pm = 1
                elif hold_time == "eight pm":
                    eight_pm = 1
                elif hold_time == "nine pm":
                    nine_pm = 1
                elif hold_time == "ten pm":
                    ten_pm = 1
                elif hold_time == "eleven pm":
                    eleven_pm = 1
                elif hold_time == "twelve am":
                    twelve_am = 1
                else:
                    one_am = 1

                ##satisfies if peak or not peak
                if sat == 1 or sun == 1:
                    peak = 0
                else:
                    peak = determine_peak(0, pta)

                """
                        In other experiments Vector x added the features  'peak', and 'weekend', and 'monday''tues'
                        'wednesday' 'thursday' 'friday' to determine if the inclusion of these days had an impact
                        determine if there was an impact
                          """
                vector_x.append([extra_weight_no_delay, hold_clem_delay, hold_liverpool_delay, hold_colch_delay, delay_arr, mon, tues, wed, thur, fri, sat, sun, six, seven, eight, nine, ten, eleven, twelve, one_pm, two_pm, three_pm, four_pm, five_pm, six_pm, seven_pm, eight_pm, nine_pm, ten_pm, eleven_pm, twelve_am, one_am])
                index = 1 + index
    return vector_y, vector_x


def sortTimes_liverpool(df):
    """
                 Test the MLP's accuracy in predicting the departing delay at each station by checking for
                a discrepancy between planned departure time (ptd) and actual departure time(adt).
                     Args:
                         df(obj): a DataFrame object that reads the csv file

                     Returns:
                         vector_x(obj): a matrix that holds different features
                         vector_y(obj): a matrix that holds a 1 for delay or 0 for no delay

                           """
    six, seven, eight, nine, ten, eleven, twelve, one_pm, two_pm, three_pm, four_pm, five_pm, six_pm, seven_pm, eight_pm, nine_pm, ten_pm, eleven_pm, twelve_am, one_am = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    vector_x = []
    vector_y = []
    mon_avg = 0
    tues_avg = 0
    wed_avg = 0
    thur_avg = 0
    fri_avg = 0
    sun_avg = 0
    sat_avg = 0
    more_likely_60 = 0
    index = 0
    weekend = 0
    delay = 0 ##holds whether or not there is a delay
    df = df.loc[df['tpl'] == 'LIVST']
    for row in df.itertuples():
        weekend = 0
        mon, tues, wed, thur, fri, sat, sun = (0, 0, 0, 0, 0, 0, 0)
        if row.ptd == '\\N' or  row.dep_at == '\\N':
            pass
        else:
            ##satisfies what day it is digit
            mon, tues, wed, thur, fri, sat, sun = find_day(row)     ##print("mon: " + str(mon) + " tues: " + str(tues) + " wed: " + str(wed) + " thurs: " + str(thur) + " fri: " + str(fri))
            ptd = datetime.strptime(row.ptd, '%H:%M').time()
            dep_at = datetime.strptime(row.dep_at, '%H:%M').time()
            # print(ptd)

            ##get if delay, and get exact delay
            start_dt = datetime.strptime(row.ptd, '%H:%M')
            end_dt = datetime.strptime(row.dep_at, '%H:%M')
            diff = (end_dt - start_dt)
            diff2 = (start_dt - end_dt)
            if (diff.seconds < diff2.seconds):      ##convert between days
                final_delay = diff.seconds
            else:
                final_delay = diff2.seconds
            ##print("delay " + str(final_delay))

            ##figure out of there is a delay
            if final_delay > 0:
                delay = 1
            else:
                delay = 0
                extra_weight_no_delay = 0

            vector_y.append(final_delay)

            ##satisfies what time it is
            hold_time = determine_time(ptd)
            if hold_time == "six":
                six = 1
            elif hold_time == "seven":
                seven = 1
            elif hold_time == "eight":
                eight = 1
            elif hold_time == "nine":
                nine = 1
            elif hold_time == "ten":
                ten = 1
            elif hold_time == "eleven":
                eleven = 1
            elif hold_time == "twelve":
                twelve = 1
            elif hold_time == "one pm":
                one_pm = 1
            elif hold_time == "two pm":
                two_pm = 1
            elif hold_time == "three pm":
                three_pm = 1
            elif hold_time == "four pm":
                four_pm = 1
            elif hold_time == "five pm":
                five_pm = 1
            elif hold_time == "six pm":
                six_pm = 1
            elif hold_time == "seven pm":
                seven_pm = 1
            elif hold_time == "eight pm":
                eight_pm = 1
            elif hold_time == "nine pm":
                nine_pm = 1
            elif hold_time == "ten pm":
                ten_pm = 1
            elif hold_time == "eleven pm":
                eleven_pm = 1
            elif hold_time == "twelve am":
                twelve_am = 1
            else:
                one_am = 1

            if mon == 1 and delay == 1:
                mon_avg =  125.31976744186046
            if tues == 1 and delay == 1:
                tues_avg = 130.07874015748033
            if wed == 1 and delay == 1:
                wed_avg = 155.94713656387665
            if thur == 1and delay == 1:
                thur_avg = 170.3673469387755
            if fri == 1and delay == 1:
                fri_avg = 146.72131147540983
            if sat == 1and delay == 1:
                sat_avg = 83.05369127516778
            if sun == 1and delay == 1:
                sun_avg = 89.25465838509317

            ##satisfies if peak or not peak
            if sat == 1 or sun == 1:
                peak = 0
                weekend = (sat_avg + sun_avg) / 2
            else:
                peak = determine_peak(0, ptd)

            """ The features that were commented out below were used previously and determined to
                         have a worse impact on the model"""
            # final_delay_arr, tues_avg, mon_avg, wed_avg, thur_avg, fri_avg, sun_avg,

            vector_x.append([weekend, extra_weight_no_delay, peak, delay, mon, tues, wed, thur, fri, sat, sun, six, seven, eight, nine, ten, eleven, twelve, one_pm, two_pm, three_pm, four_pm, five_pm, six_pm, seven_pm, eight_pm, nine_pm, ten_pm, eleven_pm, twelve_am, one_am])
            index = 1 + index
    return vector_y, vector_x

def sortTimes_chlmsfield(df, liverpool_delay):
    """
                    Test the MLP's accuracy in predicting the departing delay at each station by checking for
                   a discrepancy between planned departure time (ptd) and actual departure time(adt).
                        Args:
                            df(obj): a DataFrame object that reads the csv file
                            liverpool_delay(obj): a list that holds the liverpool delay amount

                        Returns:
                            vector_x(obj): a matrix that holds different features
                            vector_y(obj): a matrix that holds a 1 for delay or 0 for no delay

                              """
    six, seven, eight, nine, ten, eleven, twelve, one_pm, two_pm, three_pm, four_pm, five_pm, six_pm, seven_pm, eight_pm, nine_pm, ten_pm, eleven_pm, twelve_am, one_am = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    vector_x = []
    vector_y = []
    thur_avg, fri_avg, sat_avg, sun_avg = 0
    #hold_liverpool_delay = liverpool_delay  commented out to get individual predictions, remove comment to build off previous

    extra_weight_no_delay = 0
    df = df.loc[df['tpl'] == 'CHLMSFD']

    more_likely_60 = 0

    for row in df.itertuples():
        weekend = 0
        mon, tues, wed, thur, fri, sat, sun = (0, 0, 0, 0, 0, 0, 0)
        if not liverpool_delay:
            pass
        else:
            hold_liverpool_delay = liverpool_delay.pop(0)
        if row.ptd == '\\N' or  row.dep_at == '\\N':
            pass
        elif row.pta == '\\N' or  row.arr_at == '\\N':
            pass
        else:
            ##satisfies what day it is digit
            mon, tues, wed, thur, fri, sat, sun = find_day(row)     ##print("mon: " + str(mon) + " tues: " + str(tues) + " wed: " + str(wed) + " thurs: " + str(thur) + " fri: " + str(fri))
            ptd = datetime.strptime(row.ptd, '%H:%M').time()
            dep_at = datetime.strptime(row.dep_at, '%H:%M').time()
            # print(ptd)

            ##get if delay, and get exact delay
            start_dt = datetime.strptime(row.ptd, '%H:%M')
            end_dt = datetime.strptime(row.dep_at, '%H:%M')
            diff = (end_dt - start_dt)
            diff2 = (start_dt - end_dt)
            if (diff.seconds < diff2.seconds):      ##convert between days
                final_delay = diff.seconds
            else:
                final_delay = diff2.seconds

            if final_delay > 0:
                delay = 1
            else:
                delay = 0
                extra_weight_no_delay = 0
            # ##see if there is a delay at liverpool station
            # start_dt_liv = datetime.strptime(second_row.ptd, '%H:%M')
            # end_dt_liv = datetime.strptime(second_row.dep_at, '%H:%M')
            # diff_liv = (end_dt_liv - start_dt_liv)
            # diff2_liv = (start_dt_liv - end_dt_liv)
            # if (diff_liv.seconds < diff2_liv.seconds):  ##convert between days
            #     final_delay_liv = diff_liv.seconds
            # else:
            #     final_delay_liv = diff2_liv.seconds
            #
            # ##figure out of there is a delay
            # if final_delay_liv > 0:
            #     delay_liv = 1

            ##get if delay, and get exact delay for arrival
            start_dt_arr = datetime.strptime(row.pta, '%H:%M')
            end_dt_arr = datetime.strptime(row.arr_at, '%H:%M')
            diff_arr = (end_dt_arr - start_dt_arr)
            diff2_arr = (start_dt_arr - end_dt_arr)
            if (diff_arr.seconds < diff2_arr.seconds):  ##convert between days
                final_delay_arr = diff_arr.seconds
            else:
                final_delay_arr = diff2_arr.seconds
                ##print("delay " + str(final_delay))

                ##figure out of there is a delay
            if final_delay_arr > 0:
                delay_arr = 1

            vector_y.append(final_delay)
            ## vector_y.append(final_delay_arr)

            ##satisfies what time it is
            hold_time = determine_time(ptd)
            if hold_time == "six":
                six = 1
            elif hold_time == "seven":
                seven = 1
            elif hold_time == "eight":
                eight = 1
            elif hold_time == "nine":
                nine = 1
            elif hold_time == "ten":
                ten = 1
            elif hold_time == "eleven":
                eleven = 1
            elif hold_time == "twelve":
                twelve = 1
            elif hold_time == "one pm":
                one_pm = 1
            elif hold_time == "two pm":
                two_pm = 1
            elif hold_time == "three pm":
                three_pm = 1
            elif hold_time == "four pm":
                four_pm = 1
            elif hold_time == "five pm":
                five_pm = 1
            elif hold_time == "six pm":
                six_pm = 1
            elif hold_time == "seven pm":
                seven_pm = 1
            elif hold_time == "eight pm":
                eight_pm = 1
            elif hold_time == "nine pm":
                nine_pm = 1
            elif hold_time == "ten pm":
                ten_pm = 1
            elif hold_time == "eleven pm":
                eleven_pm = 1
            elif hold_time == "twelve am":
                twelve_am = 1
            else:
                one_am = 1



            if mon == 1 and delay == 1:
                mon_avg = 217.66
            if tues == 1 and delay == 1:
                tues_avg = 252.93
            if wed == 1 and delay == 1:
                wed_avg = 284.25
            if thur == 1and delay == 1:
                thur_avg == 266.34
            if fri == 1and delay == 1:
                fri_avg == 283.13
            if sat == 1and delay == 1:
                sat_avg == 205.5
            if sun == 1and delay == 1:
                sun_avg == 137.77

            ##satisfies if peak or not peak
            if sat == 1 or sun == 1:
                peak = 0
                weekend = (sat_avg + sun_avg) / 2
            else:
                peak = determine_peak(0, ptd)

            """ The features that were commented out below were used previously and determined to
                         have a worse impact on the model"""
            # final_delay_arr, tues_avg, mon_avg, wed_avg, thur_avg, fri_avg, sun_avg,

            vector_x.append([weekend, final_delay_arr, extra_weight_no_delay, peak, delay, mon, tues, wed, thur, fri, sat, sun, six, seven, eight, nine, ten, eleven, twelve, one_pm, two_pm, three_pm, four_pm, five_pm, six_pm, seven_pm, eight_pm, nine_pm, ten_pm, eleven_pm, twelve_am, one_am])
    return vector_y, vector_x

def sortTimes_colchester(df):
    """
                    Test the MLP's accuracy in predicting the departing delay at each station by checking for
                   a discrepancy between planned departure time (ptd) and actual departure time(adt).
                        Args:
                            df(obj): a DataFrame object that reads the csv file

                        Returns:
                            vector_x(obj): a matrix that holds different features
                            vector_y(obj): a matrix that holds a 1 for delay or 0 for no delay

                              """
    six, seven, eight, nine, ten, eleven, twelve, one_pm, two_pm, three_pm, four_pm, five_pm, six_pm, seven_pm, eight_pm, nine_pm, ten_pm, eleven_pm, twelve_am, one_am = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    vector_x = []
    vector_y = []
    index = 0
    thur_avg, fri_avg, sat_avg, sun_avg = 0

    extra_weight_no_delay = 0
    df = df.loc[df['tpl'] == 'CLCHSTR']


    for row in df.itertuples():

        mon, tues, wed, thur, fri, sat, sun = (0, 0, 0, 0, 0, 0, 0)
        weekend = 0
        delay_arr = 0

        if row.ptd == '\\N' or  row.dep_at == '\\N':
            pass
        elif row.pta == '\\N' or  row.arr_at == '\\N':
            pass
        else:
            ##satisfies what day it is digit
            mon, tues, wed, thur, fri, sat, sun = find_day(row)     ##print("mon: " + str(mon) + " tues: " + str(tues) + " wed: " + str(wed) + " thurs: " + str(thur) + " fri: " + str(fri))
            ptd = datetime.strptime(row.ptd, '%H:%M').time()
            dep_at = datetime.strptime(row.dep_at, '%H:%M').time()
            # print(ptd)

            ##get if delay, and get exact delay
            start_dt = datetime.strptime(row.ptd, '%H:%M')
            end_dt = datetime.strptime(row.dep_at, '%H:%M')
            diff = (end_dt - start_dt)
            diff2 = (start_dt - end_dt)
            if (diff.seconds < diff2.seconds):      ##convert between days
                final_delay = diff.seconds
            else:
                final_delay = diff2.seconds

            if final_delay > 0:
                delay = 1
            else:
                delay = 0
                extra_weight_no_delay = 0
            # ##see if there is a delay at liverpool station
            # start_dt_liv = datetime.strptime(second_row.ptd, '%H:%M')
            # end_dt_liv = datetime.strptime(second_row.dep_at, '%H:%M')
            # diff_liv = (end_dt_liv - start_dt_liv)
            # diff2_liv = (start_dt_liv - end_dt_liv)
            # if (diff_liv.seconds < diff2_liv.seconds):  ##convert between days
            #     final_delay_liv = diff_liv.seconds
            # else:
            #     final_delay_liv = diff2_liv.seconds
            #
            # ##figure out of there is a delay
            # if final_delay_liv > 0:
            #     delay_liv = 1

            ##get if delay, and get exact delay for arrival
            start_dt_arr = datetime.strptime(row.pta, '%H:%M')
            end_dt_arr = datetime.strptime(row.arr_at, '%H:%M')
            diff_arr = (end_dt_arr - start_dt_arr)
            diff2_arr = (start_dt_arr - end_dt_arr)
            if (diff_arr.seconds < diff2_arr.seconds):  ##convert between days
                final_delay_arr = diff_arr.seconds
            else:
                final_delay_arr = diff2_arr.seconds
                ##print("delay " + str(final_delay))

                ##figure out of there is a delay
            if final_delay_arr > 0:
                delay_arr = 1

            vector_y.append(final_delay)
            ## vector_y.append(final_delay_arr)

            ##satisfies what time it is
            hold_time = determine_time(ptd)
            if hold_time == "six":
                six = 1
            elif hold_time == "seven":
                seven = 1
            elif hold_time == "eight":
                eight = 1
            elif hold_time == "nine":
                nine = 1
            elif hold_time == "ten":
                ten = 1
            elif hold_time == "eleven":
                eleven = 1
            elif hold_time == "twelve":
                twelve = 1
            elif hold_time == "one pm":
                one_pm = 1
            elif hold_time == "two pm":
                two_pm = 1
            elif hold_time == "three pm":
                three_pm = 1
            elif hold_time == "four pm":
                four_pm = 1
            elif hold_time == "five pm":
                five_pm = 1
            elif hold_time == "six pm":
                six_pm = 1
            elif hold_time == "seven pm":
                seven_pm = 1
            elif hold_time == "eight pm":
                eight_pm = 1
            elif hold_time == "nine pm":
                nine_pm = 1
            elif hold_time == "ten pm":
                ten_pm = 1
            elif hold_time == "eleven pm":
                eleven_pm = 1
            elif hold_time == "twelve am":
                twelve_am = 1
            else:
                one_am = 1


            ##satisfies if peak or not peak
            if sat == 1 or sun == 1:
                peak = 0
                weekend = 1
            else:
                peak = determine_peak(0, ptd)

            if mon == 1 and delay == 1:
                mon_avg = 319.0
            if tues == 1 and delay == 1:
                tues_avg = 338.33648393194704
            if wed == 1 and delay == 1:
                wed_avg = 316.46753246753246
            if thur == 1and delay == 1:
                thur_avg == 338.8442703232125
            if fri == 1and delay == 1:
                fri_avg == 343.65364308342134
            if sat == 1and delay == 1:
                sat_avg == 220.9090909090909
            if sun == 1and delay == 1:
                sun_avg == 265.73394495412845

            """ The features that were commented out below were used previously and determined to
             have a worse impact on the model"""
            # final_delay_arr, tues_avg, mon_avg, wed_avg, thur_avg, fri_avg, sun_avg,

            vector_x.append([ final_delay_arr, delay_arr, extra_weight_no_delay, peak, delay, mon, tues, wed, thur, fri, sat, sun, six, seven, eight, nine, ten, eleven, twelve, one_pm, two_pm, three_pm, four_pm, five_pm, six_pm, seven_pm, eight_pm, nine_pm, ten_pm, eleven_pm, twelve_am, one_am])
            index = 1 + index
    return vector_y, vector_x

def sortTimes_norwich(df):
    """
                    Test the MLP's accuracy in predicting the departing delay at each station by checking for
                   a discrepancy between planned departure time (ptd) and actual departure time(adt).
                        Args:
                            df(obj): a DataFrame object that reads the csv file

                        Returns:
                            vector_x(obj): a matrix that holds different features
                            vector_y(obj): a matrix that holds a 1 for delay or 0 for no delay

                              """
    six, seven, eight, nine, ten, eleven, twelve, one_pm, two_pm, three_pm, four_pm, five_pm, six_pm, seven_pm, eight_pm, nine_pm, ten_pm, eleven_pm, twelve_am, one_am = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    vector_x = []
    vector_y = []
    index = 0

    extra_weight_no_delay = 0
    df = df.loc[df['tpl'] == 'NRCH']


    for row in df.itertuples():

        mon, tues, wed, thur, fri, sat, sun = (0, 0, 0, 0, 0, 0, 0)
        weekend = 0
        delay_arr = 0

        if row.pta == '\\N' or  row.arr_at == '\\N':
            pass
        else:
            ##satisfies what day it is digit
            mon, tues, wed, thur, fri, sat, sun = find_day(row)     ##print("mon: " + str(mon) + " tues: " + str(tues) + " wed: " + str(wed) + " thurs: " + str(thur) + " fri: " + str(fri))

            ##get if delay, and get exact delay for arrival
            start_dt_arr = datetime.strptime(row.pta, '%H:%M')
            end_dt_arr = datetime.strptime(row.arr_at, '%H:%M')
            diff_arr = (end_dt_arr - start_dt_arr)
            diff2_arr = (start_dt_arr - end_dt_arr)
            if (diff_arr.seconds < diff2_arr.seconds):  ##convert between days
                final_delay_arr = diff_arr.seconds
            else:
                final_delay_arr = diff2_arr.seconds
                ##print("delay " + str(final_delay))

                ##figure out of there is a delay
            if final_delay_arr > 0:
                delay_arr = 1

            arr_p_row = datetime.strptime(row.pta, '%H:%M').time()

            vector_y.append(final_delay_arr)

            ##satisfies what time it is
            hold_time = determine_time(arr_p_row)
            if hold_time == "six":
                six = 1
            elif hold_time == "seven":
                seven = 1
            elif hold_time == "eight":
                eight = 1
            elif hold_time == "nine":
                nine = 1
            elif hold_time == "ten":
                ten = 1
            elif hold_time == "eleven":
                eleven = 1
            elif hold_time == "twelve":
                twelve = 1
            elif hold_time == "one pm":
                one_pm = 1
            elif hold_time == "two pm":
                two_pm = 1
            elif hold_time == "three pm":
                three_pm = 1
            elif hold_time == "four pm":
                four_pm = 1
            elif hold_time == "five pm":
                five_pm = 1
            elif hold_time == "six pm":
                six_pm = 1
            elif hold_time == "seven pm":
                seven_pm = 1
            elif hold_time == "eight pm":
                eight_pm = 1
            elif hold_time == "nine pm":
                nine_pm = 1
            elif hold_time == "ten pm":
                ten_pm = 1
            elif hold_time == "eleven pm":
                eleven_pm = 1
            elif hold_time == "twelve am":
                twelve_am = 1
            else:
                one_am = 1


            ##satisfies if peak or not peak
            if sat == 1 or sun == 1:
                peak = 0
                weekend = 1
            else:
                peak = determine_peak(0, arr_p_row)

            # if mon == 1 and delay == 1:
            #     mon_avg = 319.0
            # if tues == 1 and delay == 1:
            #     tues_avg = 338.33648393194704
            # if wed == 1 and delay == 1:
            #     wed_avg = 316.46753246753246
            # if thur == 1and delay == 1:
            #     thur_avg == 338.8442703232125
            # if fri == 1and delay == 1:
            #     fri_avg == 343.65364308342134
            # if sat == 1and delay == 1:
            #     sat_avg == 220.9090909090909
            # if sun == 1and delay == 1:
            #     sun_avg == 265.73394495412845

            """ The features that were commented out below were used previously and determined to
                         have a worse impact on the model"""
            # final_delay_arr, tues_avg, mon_avg, wed_avg, thur_avg, fri_avg, sun_avg,

            vector_x.append([delay_arr, extra_weight_no_delay, peak, mon, tues, wed, thur, fri, sat, sun, six, seven, eight, nine, ten, eleven, twelve, one_pm, two_pm, three_pm, four_pm, five_pm, six_pm, seven_pm, eight_pm, nine_pm, ten_pm, eleven_pm, twelve_am, one_am])
            index = 1 + index

    return vector_y, vector_x

def sortTimes_get_average(df):
    """
                    Gets the average departure delay for each day and use as a feature for the MLP classifier
                        Args:
                            df(obj): a DataFrame object that reads the csv file

                        Returns:
                            vector_x(obj): a matrix that holds different features
                            vector_y(obj): a matrix that holds a 1 for delay or 0 for no delay

                              """
    vector_x = []
    vector_y = []
    delay_mon = 0
    delay_tues = 0
    delay_wed = 0
    delay_thur = 0
    delay_fri = 0
    delay_sat = 0
    delay_sun = 0
    mon_count = 0
    tues_count = 0
    wed_count = 0
    thur_count = 0
    fri_count = 0
    sat_count = 0
    sun_count = 0

    df = df.loc[df['tpl'] == 'CHLMSFD']

    for row in df.itertuples():

        mon, tues, wed, thur, fri, sat, sun = (0, 0, 0, 0, 0, 0, 0)
        if row.ptd == '\\N' or row.dep_at == '\\N':
            pass
        else:
            mon, tues, wed, thur, fri, sat, sun = find_day(
                row)  ##print("mon: " + str(mon) + " tues: " + str(tues) + " wed: " + str(wed) + " thurs: " + str(thur) + " fri: " + str(fri))
            ptd = datetime.strptime(row.ptd, '%H:%M').time()
            dep_at = datetime.strptime(row.dep_at, '%H:%M').time()
            # print(ptd)

            ##get if delay, and get exact delay
            start_dt = datetime.strptime(row.ptd, '%H:%M')
            end_dt = datetime.strptime(row.dep_at, '%H:%M')
            diff = (end_dt - start_dt)
            diff2 = (start_dt - end_dt)
            if (diff.seconds < diff2.seconds):  ##convert between days
                final_delay = diff.seconds
            else:
                final_delay = diff2.seconds

            vector_y.append(final_delay)

            if mon == 1 and final_delay > 0:
                delay_mon = final_delay + delay_mon
                mon_count = mon_count + 1
            if tues == 1and final_delay > 0:
                delay_tues = final_delay + delay_tues
                tues_count = tues_count + 1
            if wed == 1and final_delay > 0:
                delay_wed = final_delay + delay_wed
                wed_count = wed_count + 1
            if thur == 1and final_delay > 0:
                delay_thur = final_delay + delay_thur
                thur_count = thur_count + 1
            if fri == 1and final_delay > 0:
                delay_fri = final_delay + delay_fri
                fri_count = fri_count + 1
            if sat == 1and final_delay > 0:
                delay_sat = final_delay + delay_sat
                sat_count = sat_count + 1
            if sun == 1and final_delay > 0:
                delay_sun = final_delay + delay_sun
                sun_count = sun_count + 1
    mon_avg = delay_mon / mon_count

    '''print("monday average: " + str(mon_avg))
    print("tues average: " + str(delay_tues / tues_count))
    print("wed average: " + str(delay_wed / wed_count))
    print("thur average: " + str(delay_thur / thur_count))
    print("fri average: " + str(delay_fri / fri_count))
    print("sat average: " + str(delay_sat / sat_count))
    print("sun average: " + str(delay_sun / sun_count)) '''
    return vector_y
def main():

    df = pandas.read_csv('location_2018_2.csv', names = ['rid', 'tsid', 'tpl', 'wta', 'wtp', 'wtd', 'pta', 'ptd', 'arr_et', 'arr_wet', 'arr_at', 'arr_atRemoved', 'ps_et', 'ps_wet', 'ps_at', 'ps_atRemoved', 'dep_et', 'dep_wet', 'dep_at', 'dep_atRemoved'], dtype={'rid': str})
    # sortTimes_get_average(df) #used to printout the average time delay per day to be used in the sorttimes models

    """The code below is used for correctly predicting delays using the Norwich model that inherits the other models"""
    hold_delay = liverpool_model(df)
    blah, hold, chlem_delay = chlmsfield_model(df, hold_delay)
    blah, hold, colch_delay = colchester_model(df, hold_delay, chlem_delay)
    vector_y, vector_x = norwich_model(df, hold_delay, chlem_delay, colch_delay)

    """The commented out code below is used for predicting delays in each individual model 
    using the average time delay found in each model"""

    # vector_y, vector_x = sortTimes_liverpool(df)
    # vector_y, vector_x = sortTimes_chlmsfield(df)
    # vector_y, vector_x = sortTimes_colchester(df)
    # vector_y, vector_x = sortTimes_norwich(df)

    # # print(str(vector_y))
    # # print(str(vector_x))
    perceptron(vector_x, vector_y)

if __name__ == '__main__':
    main()
