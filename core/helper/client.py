import random
from random import randint
class Client(object):

    def __init__(self, hostId, clientId, speed, category, args, traces=None):
        self.hostId = hostId
        self.clientId = clientId
        #Ahmed - based on the format from the device trace file key:432802 val:{'computation': 162.0, 'communication': 5648.109619499134}
        self.compute_speed = speed['computation']
        self.bandwidth = speed['communication']
        self.category = category
        self.args = args
        self.traces = traces
        self.battery=0
        self.batterystatus = 'alive'
        self.score = 0
        self.behavior_index = 0
        self.count=0
        self.initialbattery=0


    def getScore(self):
        return self.score

    def registerReward(self, reward):
        self.score = reward

    #TODO: clarify this part on the use of the trace!
    #Ahmed - the trace pickle file contains only 107,749 clients!
    #Format- key:3834 val:{'duration': 211625, 'inactive': [65881, 133574, 208292, 276575, 295006, 356236, 400906, 475099], 'finish_time': 518400, 'active': [12788, 100044, 188992, 271372, 276663, 352625, 356267, 441193], 'model': 'CPH1801'}
    def isAlive(self, roundduration):  # Checking left battery level for current round

        if(self.category=='H'):
            timeleft = self.battery/self.args.pow_high #in hours
            timeleft*=(60 * 60)  #time left in seconds
            if(timeleft<roundduration):
                self.setbatterystatus('dead')
                return False
            else:
                return True

        if (self.category == 'M'):
           timeleft = self.battery / self.args.pow_mid # inhours
           timeleft *= (60 * 60)  # time left in seconds
           if (timeleft < roundduration):
               self.setbatterystatus('dead')
               return False
           else:
               return True

        if (self.category == 'L'):
            timeleft = self.battery / self.args.pow_low  # inhours
            timeleft *= (60 * 60)  # time left in seconds
            if (timeleft < roundduration):
                   self.setbatterystatus('dead')
                   return False
            else:
                   return True


    #Ahmed - return the availability windows of the client
    def availabilityPeriods(self):
        period_list=[]
        for i in range(len(self.traces['inactive'])):
            period_list.append((self.traces['active'][i], self.traces['inactive'][i]))
        return period_list

    # Ahmed - NOT NEEDED - change isActive to return the index of behaviour profile of the client
    # def isActive(self, cur_time):
    #     if self.traces is None:
    #         return True
    #
    #     norm_time = cur_time % self.traces['finish_time']
    #     if norm_time > self.traces['inactive'][self.behavior_index]:
    #         self.behavior_index += 1
    #     self.behavior_index %= len(self.traces['active'])
    #     state = False
    #     if (self.traces['active'][self.behavior_index] <= norm_time <= self.traces['inactive'][self.behavior_index]):
    #         state = True
    #     return state, self.behavior_index

    #TODO clarify the contents of the device compute trace
    #Ahmed - the trace pickle file contains only 500,000 clients!
    #Format - key:432802 val:{'computation': 162.0, 'communication': 5648.109619499134}
    def getCompletionTime(self, batch_size, upload_epoch, upload_size, download_size, augmentation_factor=3.0):
        """
           Computation latency: compute_speed is the inference latency of models (ms/sample). As reproted in many papers,
                                backward-pass takes around 2x the latency, so we multiple it by 3x;
           Communication latency: communication latency = (pull + push)_update_size/bandwidth;
        """
        #return (3.0 * batch_size * upload_epoch/float(self.compute_speed) + model_size/float(self.bandwidth))
        return {'computation':augmentation_factor * batch_size * upload_epoch*float(self.compute_speed)/1000., \
                'communication': (upload_size+download_size)/float(self.bandwidth)}
        # return (augmentation_factor * batch_size * upload_epoch*float(self.compute_speed)/1000. + \
        #         (upload_size+download_size)/float(self.bandwidth))

    def getEnergy(self, comp, comm):

        comp /= (60 * 60) #convert time from seconds to hours
        comm /= (60 * 60) #convert time from seconds to hours

        # computational energy
        # GFXBench Manhattan 3.1 Offscreen Power Efficiency (System Active Power)
        if (self.category == 'H'):
            Ecomp= self.args.pow_high * (comp)
        if (self.category == 'M'):
            Ecomp= self.args.pow_mid * (comp)
        if (self.category == 'L'):
            Ecomp= self.args.pow_low * (comp)

        ''' COMMUNICATIONAL ENERGY'''

        """ Energy consumption function with respect to the elapsed time"""
        # https://ieeexplore.ieee.org/document/6240745
        # Energy Consumption in Android Phones when using Wireless Communication Technologies

        if (comm > 13.8):
            Ecomm = (35.9 * comm) + 1.58            # 3G
        else:
            Ecomm = (19.66 * comm)+ 1.425           # Wi-Fi

        Energy = Ecomp + ((Ecomm/100) * (1230.0 * 5 / 1000))
        Energy  *= self.args.energy_incr
        return Energy


    def updateutility(self, energy):

        self.battery = self.battery - energy

        return self.battery

    def getEnergyeff(self, energy, batch_size):
            efficiency = batch_size / energy    #performance per watt (samples processed per watt)
            return efficiency

    def setbatterylevel(self, category):

        #battery in percentage
        if(category=='H'):
            perc = random.randint(self.args.high_L, self.args.high_H)
            self.battery= self.args.battery_high * (perc/100)

        if (category == 'M'):
            perc = random.randint(self.args.mid_L, self.args.mid_H)
            self.battery = self.args.battery_mid * (perc/100)

        if(category == 'L'):
           perc = random.randint(self.args.mid_L, self.args.low_H)
           self.battery = self.args.battery_low * (perc/100)

        self.battery *= 5.0 / 1000.0 #convert from mAh to Wh

        return self.battery

    def getbattery(self):
        return self.battery

    def setbatterystatus(self, status):
        self.batterystatus=status


    def getbatterystatus(self):
        if(self.batterystatus=='alive'):
            return True
        else:
            return False

    def idlepowerdeduction(self,duration):
        duration /= (60 * 60)  # convert time from seconds to hours
        #10% per hour to account for any usage
        # https://arxiv.org/ftp/arxiv/papers/1312/1312.6740.pdf
        # The Power of Smartphones
        perc = duration * self.args.usage_param
        self.battery -= self.battery * perc/100
        return self.battery

    



















