#!/usr/bin/env python3
"""ATLAS sponsor cut v2: textured lab, command center and epic intercept."""

from __future__ import annotations

import math
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

from build_atlas_hollywood_cut import ATTACK, HollywoodFilm
from build_atlas_spy_demo import CYAN, CYAN_SOFT, GREEN, RED, TYPE, WHITE, VideoSource, alpha_rect, clamp, contain, cover, paste, smooth


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/spy_demo"
WORK = OUT / "work/hollywood_cut_v2"
MONITOR_RECORDING = Path(
    "/var/folders/yj/0rnc6hws72n_zrcq_z42nzgr0000gn/T/TemporaryItems/"
    "NSIRD_screencaptureui_OLf7Sl/Screen Recording 2026-08-16 at 14.58.35.mov"
)
FULL_PATROL_RECORDING = Path(
    "/var/folders/yj/0rnc6hws72n_zrcq_z42nzgr0000gn/T/TemporaryItems/"
    "NSIRD_screencaptureui_5onHT6/Screen Recording 2026-08-16 at 20.11.00.mov"
)
ALT_RECORDING = Path(
    "/var/folders/yj/0rnc6hws72n_zrcq_z42nzgr0000gn/T/TemporaryItems/"
    "NSIRD_screencaptureui_2NjbZT/Screen Recording 2026-08-16 at 20.19.37.mov"
)
MUSIC = Path("/Users/yamromano/Downloads/Detective Background Music | Crime Scene, Spy, Investigation | Royalty Free.mp4")
SILENT = WORK / "atlas_hollywood_v2_silent.mp4"
SFX = WORK / "atlas_hollywood_v2_sfx.wav"
FINAL = OUT / "ATLAS_HOLLYWOOD_MISSION_REBUILT_2MIN.mp4"
WIDTH, HEIGHT, FPS, DURATION = 1920, 1080, 24, 120.0


def panel(frame: np.ndarray, rect: tuple[int, int, int, int], title: str, accent=CYAN) -> None:
    x0, y0, x1, y1 = rect
    cv2.rectangle(frame, (x0, y0), (x1, y1), (39, 65, 73), 1, cv2.LINE_AA)
    alpha_rect(frame, (x0, y0, x1, y0 + 42), (2, 7, 10), .90)
    cv2.line(frame, (x0, y0 + 42), (x1, y0 + 42), accent, 1, cv2.LINE_AA)
    TYPE.draw(frame, (x0 + 16, y0 + 22), title, 14, accent, "tech", anchor="lm")


class CommandFilm(HollywoodFilm):
    def __init__(self) -> None:
        super().__init__()
        if not MONITOR_RECORDING.is_file(): raise FileNotFoundError(MONITOR_RECORDING)
        if not ALT_RECORDING.is_file(): raise FileNotFoundError(ALT_RECORDING)
        if not FULL_PATROL_RECORDING.is_file(): raise FileNotFoundError(FULL_PATROL_RECORDING)
        self.monitor = VideoSource(MONITOR_RECORDING)
        self.full_patrol = VideoSource(FULL_PATROL_RECORDING)
        self.alt_monitor = VideoSource(ALT_RECORDING)
        self.attack_source = VideoSource(ATTACK)

    def full_parts(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        src = self.full_patrol.frame(seconds)
        h, w = src.shape[:2]
        mesh = src[round(h*.075):round(h*.965), 0:round(w*.723)]
        camera = src[round(h*.075):round(h*.252), round(w*.727):w]
        return mesh, camera

    def scene_geo_trace(self, local: float) -> np.ndarray:
        """A clean intelligence-opening: region to classified indoor site."""
        frame = np.full((self.h, self.w, 3), (9, 7, 4), np.uint8)
        for x in range(0, self.w, 96): cv2.line(frame,(x,0),(x,self.h),(23,31,31),1)
        for y in range(0, self.h, 96): cv2.line(frame,(0,y),(self.w,y),(23,31,31),1)
        # Abstract eastern-Mediterranean coastline—deliberately an interface,
        # not a low-resolution screenshot or a false satellite image.
        coast=np.array([(1110,110),(1070,190),(1088,265),(1038,338),(1065,412),(1015,486),(1035,565),(990,640),(1010,730),(965,820),(982,980)],np.int32)
        cv2.polylines(frame,[coast],False,(65,112,122),3,cv2.LINE_AA)
        for r in (260,175,90): cv2.circle(frame,(1112,590),r,CYAN_SOFT,1,cv2.LINE_AA)
        p=smooth(local/5.0); r=round(270*(1-p)+36)
        cv2.circle(frame,(1112,590),r,RED,2,cv2.LINE_AA)
        cv2.circle(frame,(1112,590),7,RED,-1,cv2.LINE_AA)
        cv2.line(frame,(1112-r-80,590),(1112+r+80,590),CYAN_SOFT,1)
        cv2.line(frame,(1112,590-r-80),(1112,590+r+80),CYAN_SOFT,1)
        alpha_rect(frame,(68,150,775,610),(2,6,9),.84)
        cv2.line(frame,(98,196),(310,196),CYAN,4,cv2.LINE_AA)
        TYPE.draw(frame,(98,238),"SECURE INTELLIGENCE CHANNEL",17,CYAN,"tech")
        TYPE.draw(frame,(98,305),"AN UNIDENTIFIED SIGNAL",52,WHITE,"condensed")
        TYPE.draw(frame,(98,372),"ENTERED A CONTROLLED SITE.",52,WHITE,"condensed")
        TYPE.draw(frame,(98,448),"EASTERN MEDITERRANEAN // SITE WITHHELD",16,(160,185,190),"tech")
        TYPE.draw(frame,(98,492),"CASE 07  •  SOURCE: NEO1",16,RED,"tech")
        self.cinematic_bar(frame,"GEOLOCATION TRACE")
        return frame

    def monitor_parts(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        src = self.monitor.frame(seconds)
        h, w = src.shape[:2]
        # Semantic viewport extraction: textured 3D room and its live camera.
        mesh = src[round(h*.075):round(h*.965), 0:round(w*.723)]
        # Camera pixels only. The lower source region contains an app-owned
        # "TSolve Drone Replay / results" block and must never enter the film.
        camera = src[round(h*.075):round(h*.252), round(w*.727):w]
        return mesh, camera

    def alt_parts(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        src = self.alt_monitor.frame(seconds)
        h, w = src.shape[:2]
        mesh = src[round(h*.075):round(h*.965), 0:round(w*.723)]
        camera = src[round(h*.075):round(h*.252), round(w*.727):w]
        return mesh, camera

    def textured_room(self, local: float) -> np.ndarray:
        # A single uninterrupted hero reveal of the actual textured lab.
        # Preserve the textured orbit used by the stronger previous cut.
        source = .5 + local
        mesh, _ = self.monitor_parts(source)
        frame = contain(mesh, self.w, self.h, (3, 7, 10))
        frame = cv2.convertScaleAbs(frame, alpha=.91, beta=-5)
        alpha_rect(frame, (0, 0, self.w, self.h), (1, 5, 8), .08)
        alpha_rect(frame, (52, 104, 850, 382), (2, 6, 9), .76)
        cv2.line(frame, (84, 144), (284, 144), CYAN, 4, cv2.LINE_AA)
        TYPE.draw(frame, (84, 182), "ENVIRONMENT ONLINE", 18, CYAN, "tech")
        TYPE.draw(frame, (84, 224), "THE ROOM BECOMES THE MAP.", 57, WHITE, "condensed", spacing=3, stroke=1, stroke_color=(0,0,0))
        TYPE.draw(frame, (84, 319), "A navigable spatial model—before the mission begins.", 22, (190, 210, 213), "regular")
        self.cinematic_bar(frame, "SPATIAL RECONSTRUCTION")
        self.scan(frame, local)
        return frame

    def command_center(self, local: float) -> np.ndarray:
        frame = np.full((self.h, self.w, 3), (10, 7, 3), np.uint8)
        for x in range(0, self.w, 96): cv2.line(frame, (x, 0), (x, self.h), (18, 19, 17), 1)
        for y in range(0, self.h, 96): cv2.line(frame, (0, y), (self.w, y), (18, 19, 17), 1)
        # Three genuinely different recorded viewpoints, presented as one
        # command-centre moment without repeating a source interval.
        mesh, live = self.monitor_parts(34.0 + local * 1.10)
        alt_b, live_b = self.alt_parts(4.0 + local * .55)
        alt_c, live_c = self.alt_parts(65.0 + local * .65)

        main = contain(mesh, 1110, 682, (3, 8, 11))
        paste(frame, main, 42, 94); panel(frame, (42, 94, 1152, 776), "ATLAS-01 // SITE ALPHA")
        feed1 = cover(live, 650, 305)
        paste(frame, feed1, 1218, 94); panel(frame, (1218, 94, 1868, 399), "ATLAS-01 // LIVE OPTICAL")
        feed2 = cover(alt_b, 312, 300)
        feed3 = cover(alt_c, 312, 300)
        paste(frame, feed2, 1218, 426); panel(frame, (1218, 426, 1530, 726), "ATLAS-02 // SITE BETA")
        paste(frame, feed3, 1556, 426); panel(frame, (1556, 426, 1868, 726), "ATLAS-03 // SITE GAMMA")

        for i, (name, state, color) in enumerate((("ATLAS-01", "TRACKING", GREEN), ("ATLAS-02", "PATROL", CYAN), ("ATLAS-03", "PATROL", CYAN))):
            x = 55 + i * 374
            alpha_rect(frame, (x, 817, x+340, 970), (2, 7, 10), .88)
            cv2.rectangle(frame, (x, 817), (x+340, 970), (40, 68, 76), 1)
            cv2.circle(frame, (x+28, 852), 7, color, -1, cv2.LINE_AA)
            TYPE.draw(frame, (x+49, 852), name, 17, WHITE, "tech", anchor="lm")
            TYPE.draw(frame, (x+22, 905), state, 22, color, "tech")
            TYPE.draw(frame, (x+22, 946), "POSE CONFIRMED", 13, (154,181,185), "tech")
        alpha_rect(frame, (1180, 817, 1868, 970), (2, 7, 10), .88)
        cv2.rectangle(frame, (1180, 817), (1868, 970), CYAN_SOFT, 1)
        TYPE.draw(frame, (1210, 855), "MISSION CONTROL", 17, CYAN, "tech")
        TYPE.draw(frame, (1210, 906), "3 AIRCRAFT ONLINE", 32, WHITE, "condensed")
        TYPE.draw(frame, (1210, 946), "SHARED SITUATIONAL AWARENESS", 14, GREEN, "tech")
        self.cinematic_bar(frame, "MULTI-DRONE OPERATIONS")
        return frame

    def patrol_console(self, local: float) -> np.ndarray:
        # Recreate the attached patrol style with only meaningful information.
        frame = np.full((self.h, self.w, 3), (10, 7, 3), np.uint8)
        # The newly recorded complete patrol is the narrative backbone.
        # This interval is unique and advances continuously through the route.
        mesh, live = self.full_parts(20.0 + local * 3.15)
        map_view = contain(mesh, 1322, 865, (3, 8, 11))
        camera = cover(live, 510, 287)
        paste(frame, map_view, 32, 82); panel(frame, (32, 82, 1354, 947), "LIVE PATROL // TEXTURED MAP")
        paste(frame, camera, 1380, 82); panel(frame, (1380, 82, 1890, 369), "DRONE OPTICAL FEED")
        alpha_rect(frame, (1380, 395, 1890, 947), (2, 7, 10), .92)
        cv2.rectangle(frame, (1380, 395), (1890, 947), CYAN_SOFT, 1)
        TYPE.draw(frame, (1412, 436), "ATLAS-01", 18, CYAN, "tech")
        TYPE.draw(frame, (1412, 490), "AUTONOMOUS PATROL", 31, WHITE, "condensed")
        values = (("LOCALIZATION", "CONFIRMED", GREEN), ("MISSION", "ACTIVE", CYAN), ("ROUTE", "LIVE", CYAN), ("SAFETY", "ARMED", GREEN))
        for i, (label, value, color) in enumerate(values):
            y = 548 + i*84
            cv2.line(frame, (1412, y+51), (1857, y+51), (36,57,63), 1)
            TYPE.draw(frame, (1412, y), label, 13, (137,166,171), "tech")
            TYPE.draw(frame, (1857, y+27), value, 18, color, "tech", anchor="rm")
        TYPE.draw(frame, (1412, 910), "POSITION UPDATES STREAMING", 14, GREEN, "tech")
        self.cinematic_bar(frame, "LIVE MISSION VIEW")
        return frame

    def scene_route_plan(self, local: float) -> np.ndarray:
        """Animate a route around visible safety regions on the real mesh."""
        mesh, _ = self.monitor_parts(18.0 + local*.55)
        frame = contain(mesh,self.w,self.h,(3,8,11))
        frame=cv2.convertScaleAbs(frame,alpha=.92,beta=-6)
        self.cinematic_bar(frame,"AUTONOMOUS ROUTE PLANNING")
        # Screen-space route and clearance zones are an explanatory overlay;
        # no claim is made that these pixels are raw planner telemetry.
        obstacles=[(820,420,1060,620),(1260,360,1455,560)]
        for i,(x0,y0,x1,y1) in enumerate(obstacles):
            alpha_rect(frame,(x0,y0,x1,y1),(8,18,70),.28)
            cv2.rectangle(frame,(x0,y0),(x1,y1),RED,2,cv2.LINE_AA)
            TYPE.draw(frame,(x0+12,y0+25),f"CLEARANCE ZONE {i+1}",13,RED,"tech")
        route=np.array([(410,770),(650,690),(750,575),(775,355),(1110,310),(1190,650),(1510,725)],np.int32)
        total=len(route)-1; q=clamp(local/9.0)*total; done=int(q); frac=q-done
        shown=[tuple(p) for p in route[:done+1]]
        if done<total:
            a=route[done]; b=route[done+1]; shown.append(tuple(np.rint(a+(b-a)*frac).astype(int)))
        if len(shown)>1: cv2.polylines(frame,[np.asarray(shown,np.int32)],False,CYAN,7,cv2.LINE_AA)
        for idx,pnt in enumerate(route):
            cv2.circle(frame,tuple(pnt),11,GREEN if idx<=done else CYAN_SOFT,2,cv2.LINE_AA)
        alpha_rect(frame,(54,710,650,930),(2,7,10),.78)
        TYPE.draw(frame,(84,758),"ROUTE SOLUTION",18,CYAN,"tech")
        TYPE.draw(frame,(84,815),"SAFE PATH FOUND",46,WHITE,"condensed")
        TYPE.draw(frame,(84,865),"OBSTACLE CLEARANCE VERIFIED",15,GREEN,"tech")
        TYPE.draw(frame,(84,900),"MISSION VECTOR READY",14,(160,185,190),"tech")
        return frame

    def localization_proof(self, local: float) -> np.ndarray:
        """A native-HD, information-rich self-localization chapter."""
        frame = self.attack_frame(7.0 + local * 1.45)
        frame = cv2.convertScaleAbs(frame, alpha=.96, beta=-4)
        self.cinematic_bar(frame, "LIVE SELF-LOCALIZATION")
        alpha_rect(frame, (48, 98, 595, 352), (2, 7, 10), .82)
        cv2.line(frame, (76, 137), (252, 137), CYAN, 4, cv2.LINE_AA)
        TYPE.draw(frame, (76, 174), "VISION-BASED POSE", 17, CYAN, "tech")
        TYPE.draw(frame, (76, 219), "POSITION CONFIRMED", 42, WHITE, "condensed")
        TYPE.draw(frame, (76, 283), "COLMAP + TSOLVE", 19, GREEN, "tech")
        TYPE.draw(frame, (76, 320), "LIVE FRAME AUTHORITY", 14, (153,181,185), "tech")
        # High-visibility tracked anchors and short-lived correspondence links.
        rng = np.random.default_rng(1701)
        points = []
        for i in range(58):
            x = round((.10 + rng.random()*.80)*self.w)
            y = round((.14 + rng.random()*.67)*self.h)
            phase = (local*2.4 + i*.31) % 3.0
            if phase < 2.15:
                points.append((x,y))
                cv2.circle(frame,(x,y),6,CYAN,2,cv2.LINE_AA)
                cv2.line(frame,(x-10,y),(x+10,y),CYAN_SOFT,1,cv2.LINE_AA)
                cv2.line(frame,(x,y-10),(x,y+10),CYAN_SOFT,1,cv2.LINE_AA)
        for a,b in zip(points[::7],points[3::7]):
            cv2.line(frame,a,b,(70,135,150),1,cv2.LINE_AA)
        TYPE.draw(frame,(self.w-62,100),f"{len(points):02d} VISUAL ANCHORS",15,CYAN,"tech",anchor="rm")
        # A restrained pose rail shows progress without pretending to be raw telemetry.
        x0, x1, y = 76, 740, self.h - 86
        alpha_rect(frame, (48, y-46, 780, y+31), (2,7,10), .72)
        cv2.line(frame, (x0, y), (x1, y), (50,70,76), 3, cv2.LINE_AA)
        cv2.line(frame, (x0, y), (x0+round((x1-x0)*clamp(local/13)), y), CYAN, 5, cv2.LINE_AA)
        TYPE.draw(frame, (x1, y-20), "TRACK CONTINUITY", 13, GREEN, "tech", anchor="rm")
        return frame

    def patrol_restored(self, local: float) -> np.ndarray:
        """A unique native-HD patrol interval after recovery."""
        frame = cover(self.path_b.frame(722.0 + local * 1.65), self.w, self.h)
        frame = cv2.convertScaleAbs(frame, alpha=.98, beta=-4)
        self.cinematic_bar(frame, "MISSION CONTINUES")
        alpha_rect(frame, (52, 756, 725, 925), (2,7,10), .76)
        TYPE.draw(frame, (82, 803), "POSE VERIFIED", 18, GREEN, "tech")
        TYPE.draw(frame, (82, 858), "PATROL RESTORED.", 48, WHITE, "condensed")
        TYPE.draw(frame, (82, 898), "AUTONOMOUS COMMAND FLOW ACTIVE", 13, CYAN, "tech")
        return frame

    def scene_detection_strike(self, local: float) -> np.ndarray:
        frame = self.scene_detect(local)
        if local < 1.0:
            a = 1.0 - local
            overlay = np.full_like(frame, (12, 18, 90))
            frame = cv2.addWeighted(frame, 1-a*.30, overlay, a*.30, 0)
        return frame

    def scene_approach(self, local: float) -> np.ndarray:
        """Use the real high-speed attack that culminates at the target."""
        p = clamp(local / 10.0)
        # The verified fast closing movement is 38.0-41.15s. The impact is
        # placed at its physical stop, not during the later recovery rotation.
        source = 38.0 + 3.15 * (p ** .82)
        frame = self.attack_frame(source)
        self.cinematic_bar(frame, "AUTONOMOUS RESPONSE")
        self.motion_lines(frame, local, .45 + 1.85 * p)
        alpha_rect(frame, (52, 742, 780, 930), (3, 7, 10), .74)
        TYPE.draw(frame, (82, 790), "TARGET VECTOR LOCKED", 19, RED, "tech")
        TYPE.draw(frame, (82, 845), "CLOSING DISTANCE", 48, WHITE, "condensed")
        cv2.line(frame, (82, 892), (82 + round(620 * p), 892), RED, 7, cv2.LINE_AA)
        # Stronger cinematic pursuit HUD, restricted to the attack chapter.
        cx,cy=self.w//2,self.h//2
        radius=round(250-95*p)
        cv2.circle(frame,(cx,cy),radius,RED,2,cv2.LINE_AA)
        for ang in (0,90,180,270):
            a=math.radians(ang); x0=round(cx+math.cos(a)*(radius+18)); y0=round(cy+math.sin(a)*(radius+18))
            x1=round(cx+math.cos(a)*(radius+58)); y1=round(cy+math.sin(a)*(radius+58)); cv2.line(frame,(x0,y0),(x1,y1),RED,3,cv2.LINE_AA)
        TYPE.draw(frame,(self.w-68,104),"INTERCEPT VECTOR // ARMED",15,RED,"tech",anchor="rm")
        return frame

    def scene_epic_impact(self, local: float) -> np.ndarray:
        base = self.attack_frame(41.18)
        rng = np.random.default_rng(991)
        if local < .18:
            p = local/.18
            # Violent but legible camera shake into a white-hot target flash.
            dx = round(math.sin(local*210)*18*(1-p)); dy = round(math.cos(local*173)*12*(1-p))
            base = np.roll(np.roll(base, dx, 1), dy, 0)
            heat = np.full_like(base, (25, 150, 255))
            base = cv2.addWeighted(base, .65-.45*p, heat, .35+.45*p, 0)
            cv2.circle(base, (980, 540), round(90+780*p), WHITE, max(5,round(30*(1-p))), cv2.LINE_AA)
            return base
        if local < .50:
            p=(local-.18)/.32
            frame=np.full_like(base, 255)
            cv2.circle(frame,(960,540),round(1200*p),(10,70,180),-1,cv2.LINE_AA)
            particles=np.zeros_like(frame)
            for i in range(150):
                ang=rng.uniform(0,math.tau); rad=(120+900*p)*rng.uniform(.4,1)
                x=round(960+math.cos(ang)*rad); y=round(540+math.sin(ang)*rad*.55)
                if 0<=x<self.w and 0<=y<self.h: cv2.circle(particles,(x,y),int(rng.integers(1,5)),(20,170,255),-1)
            return cv2.addWeighted(frame,.82,particles,1,0)
        if local < 1.05:
            frame=np.full_like(base,(2,3,4));
            for i in range(24):
                y=int(rng.integers(60,self.h-60)); frame[y:y+int(rng.integers(1,7))]=rng.integers(8,65)
            TYPE.draw(frame,(self.w//2,self.h//2),"SIGNAL LOST",28,(112,130,134),"tech",anchor="mm")
            return frame
        source=43.20+(local-1.05)*.55
        frame=self.attack_frame(source)
        alpha_rect(frame,(0,0,self.w,self.h),(1,5,8),max(0,.5-(local-1.05)*.4))
        return frame

    def scene_recovery_flight(self, local: float) -> np.ndarray:
        """Show the post-impact rotation/repositioning as recovery, not attack."""
        p = clamp(local / 13.0)
        source = 43.20 + (88.0 - 43.20) * (p ** .92)
        frame = self.attack_frame(source)
        status = "POSITION REACQUIRED" if local > 2.6 else "RELOCALIZING"
        color = GREEN if local > 2.6 else CYAN
        self.cinematic_bar(frame, "MISSION RECOVERY")
        cv2.circle(frame, (self.w - 66, self.h - 70), 7, color, -1, cv2.LINE_AA)
        TYPE.draw(frame, (self.w - 88, self.h - 70), status, 16, color, "tech", anchor="rm")
        if local > 4.5:
            alpha_rect(frame, (54, 754, 700, 920), (3, 7, 10), .70)
            TYPE.draw(frame, (84, 801), "AUTONOMY RESTORED", 18, GREEN, "tech")
            TYPE.draw(frame, (84, 856), "PATROL CONTINUES.", 47, WHITE, "condensed")
        return frame

    def render(self, t: float) -> np.ndarray:
        if t < 5: frame = self.scene_geo_trace(t)
        elif t < 11: frame = self.scene_intrusion(t-5)
        elif t < 16: frame = self.scene_lock(t-11)
        elif t < 27: frame = self.textured_room(t-16)
        elif t < 43: frame = self.patrol_console(t-27)
        elif t < 53: frame = self.scene_route_plan(t-43)
        elif t < 64: frame = self.localization_proof(t-53)
        elif t < 74: frame = self.command_center(t-64)
        elif t < 80: frame = self.scene_detection_strike(t-74)
        elif t < 90: frame = self.scene_approach(t-80)
        elif t < 93: frame = self.scene_epic_impact(t-90)
        elif t < 106: frame = self.scene_recovery_flight(t-93)
        elif t < 113: frame = self.patrol_restored(t-106)
        else: frame = self.scene_finale(t-113)
        frame=np.clip(frame.astype(np.float32)*self.story_vignette,0,255).astype(np.uint8)
        for y in range(0,self.h,8): frame[y:y+1]=(frame[y:y+1].astype(np.float32)*.97).astype(np.uint8)
        return frame


def make_sfx(path: Path) -> None:
    rate=48000; count=round(DURATION*rate); audio=np.zeros((count,2),np.float64); rng=np.random.default_rng(816)
    def put(start, sig, left=1.0, right=1.0):
        i=round(start*rate); n=min(len(sig),count-i)
        if n>0: audio[i:i+n,0]+=sig[:n]*left; audio[i:i+n,1]+=sig[:n]*right
    # Threat atmosphere and unmistakable detection signature.
    x=np.arange(round(13*rate))/rate
    atmosphere=(.035*np.sin(math.tau*41*x)+.018*np.sin(math.tau*57*x))*np.sin(np.pi*np.clip(x/13,0,1))**.45
    put(0,atmosphere)
    for when,freq in ((.55,920),(1.02,1160),(7.15,1480),(7.38,1960),(67.08,1240),(67.31,1820)):
        x=np.arange(round(.22*rate))/rate; sig=np.sin(math.tau*freq*x)*np.sin(np.pi*x/.22)**2*.13; put(when,sig,.95,.78)
    for when in (5,11,16,27,43,53,64,74,80,93,106,113):
        x=np.arange(round(.5*rate))/rate; env=np.sin(np.pi*x/.5)**2
        sig=(rng.standard_normal(len(x))-np.roll(rng.standard_normal(len(x)),23))*env*.038; put(max(0,when-.18),sig)
    # Riser into the strike.
    x=np.arange(round(3.2*rate))/rate; env=np.clip(x/3.2,0,1)**1.7
    riser=(rng.standard_normal(len(x))-np.roll(rng.standard_normal(len(x)),31))*env*.065
    put(86.8,riser,.8,1.0)
    # Epic layered impact: transient, low body, sub-drop, debris and long tail.
    x=np.arange(round(5.0*rate))/rate
    sub=np.sin(math.tau*(74*x-13*x*x))*np.exp(-x*1.15)
    body=np.sin(math.tau*(128*x-27*x*x))*np.exp(-x*2.6)
    crack=rng.standard_normal(len(x))*np.exp(-x*22)
    debris=(rng.standard_normal(len(x))-np.roll(rng.standard_normal(len(x)),37))*np.exp(-x*3.7)
    tail=np.sin(math.tau*38*x)*np.exp(-x*.72)
    put(90.02,.70*sub+.33*body+.16*crack+.075*debris+.12*tail,1,.94)
    for when,freq in ((91.14,540),(91.52,760),(93.35,1120)):
        x=np.arange(round(.2*rate))/rate; sig=np.sin(math.tau*freq*x)*np.sin(np.pi*x/.2)**2*.10; put(when,sig)
    pcm=np.clip(audio*32767,-32768,32767).astype('<i2'); path.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(path),'wb') as f: f.setnchannels(2); f.setsampwidth(2); f.setframerate(rate); f.writeframes(pcm.tobytes())


def main() -> None:
    WORK.mkdir(parents=True,exist_ok=True)
    if '--mux-only' not in sys.argv:
        film=CommandFilm(); writer=cv2.VideoWriter(str(SILENT),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(WIDTH,HEIGHT))
        if not writer.isOpened(): raise RuntimeError(SILENT)
        try:
            for i in range(round(DURATION*FPS)):
                if i%(FPS*5)==0: print(f'rendered {i/FPS:5.1f}/{DURATION:.1f}s',flush=True)
                writer.write(film.render(i/FPS))
        finally: writer.release()
    make_sfx(SFX)
    # The supplied detective score is the continuous musical bed. Start after
    # its black lead-in and keep SFX subordinate to the score.
    af=("[1:a]atrim=4:124,asetpts=PTS-STARTPTS,loudnorm=I=-18:LRA=9:TP=-1.5,volume=.72,afade=t=in:st=0:d=2.2,afade=t=out:st=118:d=2[m];"
        "[2:a]volume=.98[s];[m][s]amix=inputs=2:duration=first:normalize=0,alimiter=limit=.92,volume=.82[a]")
    subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-i',str(SILENT),'-i',str(MUSIC),'-i',str(SFX),'-filter_complex',af,
        '-map','0:v:0','-map','[a]','-c:v','libx264','-preset','slow','-crf','17','-pix_fmt','yuv420p','-profile:v','high','-level','4.1',
        '-c:a','aac','-b:a','256k','-ar','48000','-movflags','+faststart','-t','120',str(FINAL)],check=True)
    print(FINAL)


if __name__=='__main__': main()
