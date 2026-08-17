#!/usr/bin/env python3
"""ATLAS story v3: coherent patrol backbone and frame-matched map feeds."""
from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np

from build_atlas_hollywood_cut_v2 import CommandFilm, panel
import build_atlas_action_game_cut as action_cut
from build_atlas_spy_demo import CYAN, CYAN_SOFT, GREEN, RED, TYPE, WHITE, VideoSource, alpha_rect, clamp, contain, cover, paste

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/spy_demo"
WORK = OUT / "work/story_v3"
FINAL = OUT / "ATLAS_CINEMATIC_STORY_V3.mp4"
MUSIC = Path("/Users/yamromano/Downloads/Detective Background Music | Crime Scene, Spy, Investigation | Royalty Free.mp4")
ROOM_DARK = Path("/Users/yamromano/Desktop/ATLAS_CINEMATIC_MISSION_DEMO.mp4")
SCIFI = Path("/Users/yamromano/Downloads/spy_action_intro_10s_trimmed.mp4")
ENEMY_LAB = Path("/var/folders/yj/0rnc6hws72n_zrcq_z42nzgr0000gn/T/TemporaryItems/NSIRD_screencaptureui_hwDyJ8/Screen Recording 2026-08-16 at 20.18.12.mov")
FULL_PATROL = Path("/var/folders/yj/0rnc6hws72n_zrcq_z42nzgr0000gn/T/TemporaryItems/NSIRD_screencaptureui_5onHT6/Screen Recording 2026-08-16 at 20.11.00.mov")
WIDTH, HEIGHT, FPS, DURATION = 1920, 1080, 24, 130.0
SILENT = WORK / "story_v3_silent.mp4"
SFX = WORK / "story_v3_sfx.wav"


class StoryV3(CommandFilm):
    def __init__(self):
        action_cut.SCREEN_RECORDING = FULL_PATROL
        super().__init__()
        self.room_dark_source = VideoSource(ROOM_DARK)
        self.scifi_source = VideoSource(SCIFI)
        self.enemy_lab_source = VideoSource(ENEMY_LAB)

    def source_full(self, source: VideoSource, seconds: float) -> np.ndarray:
        return cv2.convertScaleAbs(contain(source.frame(seconds), self.w, self.h, (2,5,8)), alpha=.98, beta=-2)

    def scene_room_dark(self, local: float) -> np.ndarray:
        return self.source_full(self.room_dark_source, min(6.95, local))

    def scene_scifi_intro(self, local: float) -> np.ndarray:
        frame = self.source_full(self.scifi_source, min(10.25, local))
        # Only the ATLAS channel bar is added; the supplied animation remains intact.
        self.cinematic_bar(frame, "SECURE LOCATION ACQUISITION")
        return frame

    def scene_live_launch_v3(self, local: float) -> np.ndarray:
        # Start on the original 1920x1080 drone feed: ground, takeoff, then the
        # first forward motion.  This deliberately precedes the map reveal.
        raw = self.path_b.frame(4.5 + local*1.24)
        frame = cover(raw, self.w, self.h)
        frame = cv2.convertScaleAbs(frame, alpha=1.02, beta=-3)
        self.cinematic_bar(frame, "LIVE PATROL START")
        # Restrained live-feature marks establish localization without hiding
        # the full-resolution optical view.
        rng=np.random.default_rng(321)
        for i in range(28):
            x=int(100+rng.random()*1720); y=int(115+rng.random()*720)
            if (i+int(local*5))%4==0:
                cv2.circle(frame,(x,y),5,CYAN,1,cv2.LINE_AA)
                cv2.line(frame,(x-8,y),(x+8,y),CYAN_SOFT,1,cv2.LINE_AA)
        alpha_rect(frame,(54,760,620,914),(2,7,10),.72)
        TYPE.draw(frame,(82,805),"AIRCRAFT ONLINE",17,GREEN,"tech")
        TYPE.draw(frame,(82,861),"PATROL LAUNCHED.",44,WHITE,"condensed")
        return frame

    def scene_map_reveal_v3(self, local: float) -> np.ndarray:
        # Replace the app controls entirely with the two meaningful live views:
        # the map and the original high-resolution optical feed.
        source=20.0+local*3.0
        screen=self.full_patrol.frame(source)
        sh,sw=screen.shape[:2]
        # Stop before the app's TSolve/Drone Paths column; do not allow even a
        # sliver of those controls into the cinematic map panel.
        mesh=screen[round(sh*.075):round(sh*.965),0:round(sw*.635)]
        optical=self.path_b.frame(source)
        frame=np.full((self.h,self.w,3),(3,7,10),np.uint8)
        map_view=contain(mesh,1180,880,(3,7,10))
        optical_view=contain(optical,650,880,(3,7,10))
        paste(frame,map_view,32,104)
        paste(frame,optical_view,1238,104)
        panel(frame,(32,104,1212,984),"LIVE SPATIAL MAP")
        panel(frame,(1238,104,1888,984),"SYNCHRONIZED DRONE VIEW")
        alpha_rect(frame,(70,800,540,928),(2,7,10),.76)
        TYPE.draw(frame,(96,838),"LIVE POSE",14,CYAN,"tech")
        TYPE.draw(frame,(96,887),"POSITION CONFIRMED",29,WHITE,"condensed")
        cv2.circle(frame,(500,874),8,GREEN,-1,cv2.LINE_AA)
        self.cinematic_bar(frame,"LIVE SELF-LOCALIZATION")
        return frame

    def scene_three_aircraft_v3(self, local: float) -> np.ndarray:
        frame=np.full((self.h,self.w,3),(8,6,3),np.uint8)
        pairs=[self.full_parts(46.0+local*.72),self.full_parts(112.0+local*.64)]
        # Third aircraft uses the different no-mesh replay recorded separately.
        alt_mesh,_=self.alt_parts(38.0+local*.52)
        third_live=self.path_c.frame(214.0+local*1.15)
        pairs.append((alt_mesh,third_live))
        count=1 if local<5 else (2 if local<10 else 3)
        if count==1:
            layout=[(260,132,1400,790)]
            phrase="ONE AIRCRAFT. ONE DECISION."
        elif count==2:
            layout=[(74,156,852,748),(994,156,852,748)]
            phrase="AND ANOTHER ONE."
        else:
            layout=[(44,176,584,700),(668,176,584,700),(1292,176,584,700)]
            phrase="MULTIPLE AIRCRAFT. UNITED DECISIONS."
        for i,(x,y,w,h) in enumerate(layout):
            mesh,live=pairs[i]; top_h=round(h*.36); bottom_h=h-top_h-26
            top=cover(live,w,top_h); bottom=contain(mesh,w,bottom_h,(3,8,11))
            paste(frame,top,x,y); panel(frame,(x,y,x+w,y+top_h),f"ATLAS-0{i+1} // OPTICAL")
            paste(frame,bottom,x,y+top_h+26); panel(frame,(x,y+top_h+26,x+w,y+h),f"ATLAS-0{i+1} // MATCHED MAP")
            cv2.circle(frame,(x+14,y+h+32),7,GREEN,-1,cv2.LINE_AA)
            TYPE.draw(frame,(x+34,y+h+32),"ONLINE // SHARED RESPONSE",13,GREEN,"tech",anchor="lm")
        TYPE.draw(frame,(44,100),phrase,34,WHITE,"condensed")
        if 4.65<local<5.25 or 9.65<local<10.25:
            glow=np.full_like(frame,(55,145,175)); a=.16*(1-abs((local%5)-.0)/.35 if local%5<.35 else 0)
            frame=cv2.addWeighted(frame,1-a,glow,a,0)
        self.cinematic_bar(frame,"MULTI-AIRCRAFT OPERATIONS")
        return frame

    def scene_epic_impact(self, local: float) -> np.ndarray:
        # The strongest visible flash and the audio transient now begin on the
        # same frame at 01:36.00, followed by a short dimmed signal-loss beat.
        base=self.attack_frame(41.18); rng=np.random.default_rng(991)
        if local<.12:
            p=local/.12; hot=np.full_like(base,(25,155,255))
            frame=cv2.addWeighted(base,.25*(1-p),hot,.75+.25*p,0)
            cv2.circle(frame,(960,540),round(100+900*p),WHITE,max(4,round(26*(1-p))),cv2.LINE_AA)
            return frame
        if local<.68:
            frame=np.full_like(base,(2,3,4))
            for _ in range(22):
                y=int(rng.integers(70,self.h-70)); frame[y:y+int(rng.integers(1,6))]=rng.integers(8,52)
            TYPE.draw(frame,(self.w//2,self.h//2),"SIGNAL LOST",27,(110,130,134),"tech",anchor="mm")
            return frame
        frame=self.attack_frame(43.2+(local-.68)*.55)
        alpha_rect(frame,(0,0,self.w,self.h),(1,5,8),max(0,.55-(local-.68)*1.7))
        return frame

    def scene_final_closing_v3(self, local: float) -> np.ndarray:
        """The exact close view marked by the user, from 01:35 to impact."""
        p=clamp(local/4.0)
        source=40.86+(41.18-40.86)*(p**.9)
        frame=self.attack_frame(source)
        self.cinematic_bar(frame,"AUTONOMOUS RESPONSE")
        self.motion_lines(frame,local,.65+1.3*p)
        alpha_rect(frame,(52,742,780,930),(3,7,10),.74)
        TYPE.draw(frame,(82,790),"TARGET VECTOR LOCKED",19,RED,"tech")
        TYPE.draw(frame,(82,845),"CLOSING DISTANCE",48,WHITE,"condensed")
        cv2.line(frame,(82,892),(82+round(620*p),892),RED,7,cv2.LINE_AA)
        return frame

    def scene_enemy_lab_v3(self, local: float) -> np.ndarray:
        source = .3 + local*2.15
        raw = self.enemy_lab_source.frame(source)
        frame = contain(raw,self.w,self.h,(3,7,10))
        frame=cv2.convertScaleAbs(frame,alpha=1.03,beta=-4)
        alpha_rect(frame,(44,710,820,925),(2,7,10),.80)
        TYPE.draw(frame,(76,758),"ENEMY DRONE LAB",18,RED,"tech")
        TYPE.draw(frame,(76,816),"NEO1 PROFILE LOADED",44,WHITE,"condensed")
        TYPE.draw(frame,(76,866),"DETECTION MODEL // READY",15,GREEN,"tech")
        self.cinematic_bar(frame,"THREAT INTELLIGENCE",alert=True)
        return frame

    def render(self,t:float)->np.ndarray:
        if t < 7: frame=self.scene_intrusion(t)
        elif t < 13: frame=self.scene_lock(t-7)
        elif t < 20: frame=self.scene_room_dark(t-13)
        elif t < 30.3: frame=self.scene_scifi_intro(t-20)
        elif t < 43: frame=self.scene_live_launch_v3(t-30.3)
        elif t < 59: frame=self.scene_map_reveal_v3(t-43)
        elif t < 74: frame=self.scene_three_aircraft_v3(t-59)
        elif t < 80: frame=self.scene_enemy_lab_v3(t-74)
        elif t < 95: frame=self.scene_detection_strike((t-80)%6)
        elif t < 99: frame=self.scene_final_closing_v3(t-95)
        elif t < 100: frame=self.scene_epic_impact(t-99)
        elif t < 114: frame=self.scene_recovery_flight(t-100)
        elif t < 122: frame=self.patrol_restored(t-114)
        else: frame=self.scene_finale(t-122)
        frame=np.clip(frame.astype(np.float32)*self.story_vignette,0,255).astype(np.uint8)
        for y in range(0,self.h,8): frame[y:y+1]=(frame[y:y+1].astype(np.float32)*.97).astype(np.uint8)
        return frame


def make_sfx(path:Path):
    rate=48000; audio=np.zeros((round(DURATION*rate),2),np.float64); rng=np.random.default_rng(1708)
    def put(start,sig,l=1.,r=1.):
        i=round(start*rate); n=min(len(sig),len(audio)-i)
        if n>0: audio[i:i+n,0]+=sig[:n]*l; audio[i:i+n,1]+=sig[:n]*r
    for when,freq in ((.55,920),(1.02,1160),(7.15,1480),(7.38,1960),(80.08,1240),(80.31,1820)):
        x=np.arange(round(.22*rate))/rate; put(when,np.sin(math.tau*freq*x)*np.sin(np.pi*x/.22)**2*.11,.95,.8)
    for when in (64.0,69.0):
        x=np.arange(round(.55*rate))/rate; env=np.sin(np.pi*x/.55)**2
        sweep=np.sin(math.tau*(240+720*x)*x)*env*.075
        put(when-.18,sweep,.78,1.0)
    x=np.arange(round(3*rate))/rate; put(96,(rng.standard_normal(len(x))-np.roll(rng.standard_normal(len(x)),31))*(x/3)**1.8*.05,.82,1)
    x=np.arange(round(5*rate))/rate
    sig=.7*np.sin(math.tau*(74*x-13*x*x))*np.exp(-x*1.12)+.32*np.sin(math.tau*(128*x-27*x*x))*np.exp(-x*2.6)+.16*rng.standard_normal(len(x))*np.exp(-x*23)
    put(99.0,sig,1,.95)
    path.parent.mkdir(parents=True,exist_ok=True); pcm=np.clip(audio*32767,-32768,32767).astype('<i2')
    with wave.open(str(path),'wb') as f: f.setnchannels(2); f.setsampwidth(2); f.setframerate(rate); f.writeframes(pcm.tobytes())


def main():
    WORK.mkdir(parents=True,exist_ok=True); film=StoryV3()
    writer=cv2.VideoWriter(str(SILENT),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(WIDTH,HEIGHT))
    try:
        for i in range(round(DURATION*FPS)):
            if i%(FPS*5)==0: print(f"rendered {i/FPS:5.1f}/{DURATION:.1f}s",flush=True)
            writer.write(film.render(i/FPS))
    finally: writer.release()
    make_sfx(SFX)
    af=("[1:a]atrim=4:134,asetpts=PTS-STARTPTS,loudnorm=I=-18:LRA=9:TP=-1.5,volume=.72,afade=t=in:st=0:d=2.2,afade=t=out:st=128:d=2[m];"
        "[2:a]volume=.96[s];[m][s]amix=inputs=2:duration=first:normalize=0,alimiter=limit=.92,volume=.84[a]")
    subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-i',str(SILENT),'-i',str(MUSIC),'-i',str(SFX),'-filter_complex',af,'-map','0:v:0','-map','[a]',
        '-c:v','libx264','-preset','slow','-crf','15','-pix_fmt','yuv420p','-profile:v','high','-level','4.1','-c:a','aac','-b:a','256k','-ar','48000','-movflags','+faststart','-t',str(DURATION),str(FINAL)],check=True)
    print(FINAL)

if __name__=='__main__': main()
