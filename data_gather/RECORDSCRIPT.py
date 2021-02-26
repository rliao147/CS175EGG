import os
import sys
import time
import MalmoPython
import random
import numpy as np
from PIL import Image

width = 256
height = 256
seedfile = "/Users/RyanLiao/Desktop/Malmo-0.37.0-Mac-64bit_withBoost_Python3.6/Python_Examples/FOREST"
timelimit = 40000000
eptime = 45000
numphotos = 11000

# RUN THIS SCRIPT ON MALMO PYTHON3
# HOW IT WORKS
# - specify the location of the savefile that you want to run
# - You can usually find your savefile under Minecraft/run/saves/
# - specify the size of the image that you want above
# - specify the time you want to spend running the script (RUNS UNTIL DEATH)
#   - 30000ms gathers ~8 images

#turn clouds off

def arrToImg(array, outfile):
    array = np.array(array)
    array = array.reshape(height, width, 3)
    img = Image.fromarray(array, mode='RGB')
    img.save(outfile)
    img.close()
    return True

def generateXMLbySeed():
		missionXML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
		<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
		  <About>
			<Summary>Gather Data</Summary>
		  </About>
		  <ServerSection>
			<ServerInitialConditions>
				<Time>
                    <StartTime>1000</StartTime>
				    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
			    <Weather>clear</Weather>
			</ServerInitialConditions>
			<ServerHandlers>
				<FileWorldGenerator src="{src}" forceReset="1" destroyAfterUse="0"/>
				<ServerQuitFromTimeUp timeLimitMs="{limit}"/>
				<ServerQuitWhenAnyAgentFinishes/>
			</ServerHandlers>
		  </ServerSection>

		  <AgentSection mode="Spectator">
            <Name>IceCream</Name>
            <AgentStart>
                <Placement x="{xcoord}" y="90" z="{zcoord}" yaw="90"/>
            </AgentStart>
            <AgentHandlers>
                <VideoProducer want_depth="0" viewpoint="0">
                <Width>256</Width>
                <Height>256</Height>
                </VideoProducer>
                <ContinuousMovementCommands turnSpeedDegs="180"/>
                <ChatCommands/>
                <AgentQuitFromTimeUp timeLimitMs="{tlimit}"/>
            </AgentHandlers>
          </AgentSection>
		</Mission>
		'''
		return missionXML.format(src=seedfile, limit=timelimit, xcoord=random.randint(0,300), zcoord=random.randint(100, 350), tlimit=eptime)
agent_id = 10001
counter = 9019
while counter < numphotos:
    agent_host = MalmoPython.AgentHost()

    try:
        missionXML = generateXMLbySeed()
        my_mission = MalmoPython.MissionSpec(missionXML, True)
        my_mission_record = MalmoPython.MissionRecordSpec()
    except Exception as e:
        print("open mission ERROR: ", e)

    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available
    agent_id += 1
    # Attempt to start a mission:
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_clients, my_mission_record, 0, "IMGCOLLECTOR")
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission to start ")
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
    print("Mission running ")

    agent_host.sendCommand("pitch 0.1")
    time.sleep(0.5)
    agent_host.sendCommand("pitch 0")

    while world_state.is_mission_running:
        time.sleep(random.random())
        agent_host.sendCommand( "turn " + str(0.5*(random.random()*2-1)) )
        time.sleep(random.random())
        world_state = agent_host.getWorldState()
        agent_host.sendCommand("move 2")
        for i in range(5):
            agent_host.sendCommand("turn -1")
            agent_host.sendCommand("move 2")
            agent_host.sendCommand("turn -1")
            agent_host.sendCommand("turn 1")
            agent_host.sendCommand("move " + str(0.5 * (random.random() * 100 - 0.5)))
            agent_host.sendCommand("turn " + str(0.5 * (random.random() * 2 - 1)))
            time.sleep(0.05)
        print("saved image", counter)
        try:
            img = world_state.video_frames[-1].pixels
            arrToImg(img, "./img/" + str(counter) + ".jpg")
        except Exception as e:
            print("Error:", e)
        counter += 1

    # print "Mission running "
    print("Mission End")