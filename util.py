import sys, os, datetime
 
class Visualization:
    def __init__(self):
        os.system('')
        self.start_time = datetime.datetime.now()
 
    def get_number_digit(self, number):
        count = 1
        n = number
        n //= 10
        while n > 0:
            n //= 10
            count += 1
        return count
 
    def hilite(self, string, style=0, text_color=30, background_color=40):
        #style 0 - 8
        #text_color 30 - 37
        #backgroung_color 40 - 47
        return '\x1b[%sm%s\x1b[0m' % (';'.join([str(style), str(text_color), str(background_color)]), string)
    
    def convertation(self, c, total):
        if total < 1024:
            return (round(c, 2), round(total, 2), "B") #B
        elif total/1024 < 1024:
            return (round(c/1024, 2), round(total/1024, 2), "KB") #KB
        elif total/1024 > 1024:
            return (round(c/1024**2, 2), round(total/1024**2, 2), "MB") #MB
        elif total/1024**2 < 1024:
            return (round(c/1024**3, 2), round(total/1024**3, 2), "GB") #GB
        elif total/1024**2 > 1024:
            return (round(c/1024**3, 2), round(total/1024**3, 2), "TB") #TB
    
    def calc_eta(self, total, done, round_m=1):
        now = datetime.datetime.now()
        left = (total-done) * (now - self.start_time)/done
        return round(left.total_seconds(), round_m)
    
    def print_progress_bar(self, done, total, conv=False, eta=False, eta_i=[0,0,0], empty_cell=" ", label="", points = 30):
        if conv:
            convertated = self.convertation(done, total)
            c, e, l = convertated[0], convertated[1], convertated[2]
        else:
            c, e, l = done, total, " "
        
        try:
            current_prog = int(done*points/total)
            current_prog_perc = int(done*100/total)
        except ZeroDivisionError:
            current_prog, current_prog_perc = 0, 0
        sys.stdout.write("\r{0}/{1}{2}[{3}{4}]: {5}% {6} {7}".format(c, e, l + " "*(self.get_number_digit(e)-self.get_number_digit(c+1)),
         self.hilite("█", 0, 32, 40)*current_prog, self.hilite(empty_cell, 0, 30, 47)*(points - current_prog), current_prog_perc,
         "eta: %s s" % self.calc_eta(total, done) if eta else "", label))
        sys.stdout.flush()