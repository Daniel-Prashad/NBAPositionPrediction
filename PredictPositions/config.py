# define the list of players for each position to be used in the training and test sets
point_guards = ['steve nash', 'jason kidd', 'chris paul', 'tony parker', 'russell westbrook', 'luka doncic', 'trae young', 'ja morant', 'darius garland', 'lonzo ball',
                'damian lillard', 'derrick rose', 'jamal murray', 'kyrie irving', 'cameron payne', 'malcolm brogdon', 'fred vanvleet', 'dennis schroder', 'kemba walker', 'baron davis',
                'patrick beverley', 'mike conley', 'sam cassell', 'deron williams', 'shaun livingston', 'michael carter-williams', 'stephen curry', 'john stockton']

shooting_guards = ['allen iverson', 'ray allen', 'kobe bryant', 'tracy mcgrady','michael jordan', 'anthony edwards', 'tyler herro', 'devin booker', 'donovan mitchell', 'jordan poole',
                    'vince carter', 'demar derozan', 'klay thompson', 'terrence ross', 'cj mccollum', 'buddy hield', 'victor oladipo', 'caris levert', 'michael redd', 'richard hamilton',
                    'jason terry', 'monta ellis', 'jason richardson', 'allan houston', 'morris peterson', 'tony allen', 'ron harper', 'zach lavine']

small_forwards = ['lebron james', 'paul pierce', 'scottie pippen', 'kevin durant', 'larry bird', 'jayson tatum', 'brandon ingram', 'andrew wiggins', 'michael porter jr', 'o.g. anunoby',
                    'khris middleton', 'danny granger', 'kawhi leonard', 'robert covington', 'kelly oubre jr', 'jalen rose', 'mikal bridges', 'michael finley', 'shane battier', 'james posey', 
                    'glenn robinson', 'dillon brooks', 'carmelo anthony', 'james worthy', 'shawn marion', 'dominique wilkins', 'grant hill', 'gordon hayward']

power_forwards = ['kevin garnett', 'dirk nowitzki', 'dennis rodman', 'charles barkley', 'amar\'e stoudemire', 'pascal siakam', 'blake griffin', 'domantas sabonis', 'john collins', 'udonis haslem',
                    'draymond green', 'giannis antetokounmpo', 'zach randolph', 'kevin love', 'christian wood', 'aaron gordon', 'robert horry', 'kenyon martin', 'serge ibaka', 'tristan thompson', 
                    'elton brand', 'antawn jamison', 'james johnson', 'lamar odom', 'jerami grant', 'darius bazley', 'zion williamson', 'julius randle']

centers = ['tim duncan', 'patrick ewing', 'pau gasol', 'nikola jokic', 'dikembe mutombo', 'bam adebayo', 'jarrett allen', 'karl-anthony towns', 'deandre ayton', 'myles turner',
            'al horford', 'Kareem Abdul-Jabbar', 'demarcus cousins','marc gasol', 'andre drummond', 'nikola vucevic', 'jusuf nurkic', 'jonas valanciunas', 'mitchell robinson', 'montrezl harrell',
            'ben wallace', 'vlade divac', 'andrew bynum', 'jeff foster', 'daniel theis', 'shaquille o\'neal', 'dwight howard', 'bismack biyombo']

training_players = (point_guards + shooting_guards + small_forwards + power_forwards + centers)

testing_players = ['rajon rondo', 'john wall', 'alex caruso', 'chauncey billups', 'jrue holiday', 'ricky rubio', 'marcus smart', 'isiah thomas', 'kyle lowry', 'ben simmons', 'derek fisher', 'mike bibby',
                   'jamal crawford', 'avery bradley', 'eric bledsoe', 'austin rivers', 'norman powell', 'manu ginobili', 'jimmy butler', 'clyde drexler', 'reggie miller', 'bradley beal', 'jeremy lamb', 'james harden',
                   'trevor ariza', 'caron butler', 'andrei kirilenko', 'glen rice', 'richard jefferson', 'gerald wallace', 'tayshaun prince', 'metta world peace', 'hedo turkoglu', 'rudy gay', 'andre iguodala', 'paul george',
                   'paul millsap', 'taj gibson', 'grant williams', 'patrick patterson', 'karl malone', 'chris webber', 'thaddeus young', 'lamarcus aldridge', 'kyle kuzma', 'carlos boozer', 'andrea bargnani', 'amir johnson',
                   'moses malone', 'al jefferson', 'steven adams', 'brook lopez', 'yao ming', 'tyson chandler', 'rudy gobert', 'joel embiid', 'hakeem olajuwon', 'joakim noah', 'hassan whiteside', 'javale mcgee']

# define dictionaries to be used for mapping labels to their string representations as well as those mapping to colours to be used in scatterplots
#three_pos_dict = {0: ['GUARD', ['PG', 'SG']], 1: ['FORWARD', ['SF', 'PF']], 2: ['CENTER', ['C']]}
colours_five = {'GUARD': 'purple', 'GUARD/FORWARD': 'blue', 'FORWARD': 'green', 'FORWARD/CENTER': 'orange', 'CENTER':'red'}
#colours_three = {0: 'purple', 1: 'green', 2: 'red'}
