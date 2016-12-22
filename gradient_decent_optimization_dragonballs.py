
import random


probability_table = [
    0.10,
    0.09,
    0.08,
    0.07,
    0.06,
    0.05,
    0.04
]

try_resolution = 10000


class Gacha(object):

    def get(self):
        seed = random.random()
        index = None
        for i, score in enumerate(probability_table):
            seed -= score
            if seed <= 0.0:
                index = i
                break

        return index


class Inspector(object):

    def hasenough(self, data):
        return all(map(lambda x: x in data, range(len(probability_table))))


class Player(object):

    def want(self):
        gacha = Gacha()
        inspector = Inspector()
        try_count = 0
        results = {}
        while not inspector.hasenough(results):
            try_count += 1
            roll = gacha.get()
            if roll is None:
                continue
            if roll in results:
                results[roll] += 1
            else:
                results[roll] = 1

        return try_count


class Reporter(object):

    def count_complete(self):
        complete_count_map = {}
        for _ in range(try_resolution):
            player = Player()
            r = player.want()
            if r in complete_count_map:
                complete_count_map[r] += 1
            else:
                complete_count_map[r] = 1

        return complete_count_map

tester = Reporter()
data = tester.count_complete()

most_often_trycount = 0
most_often_occasioncount = 0
for trycount, occasioncount in data.items():
    if most_often_occasioncount < occasioncount:
        most_often_occasioncount = occasioncount
        most_often_trycount = trycount
print(most_often_trycount)
