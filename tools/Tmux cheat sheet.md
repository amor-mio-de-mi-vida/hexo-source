---
title: tmux cheat sheet
tags:
  - tools
  - tmux
categories: 
date created: 2024-10-03 21:49:51
date modified: 2024-10-03 23:17:55
---
## Session
### Start a new session
```bash
tmux
```
```vim
:new
```
### Start a new session or attach to an existing session named mysession
```bash
tmux new-session -A -s mysession
```
### Start a new session with the name mysession
```bash
tmux new -s mysession
```
```vim
:new -s mysession
```
### kill/delete all sessions but the current
```vim
:kill-session -a 
```
### kill/delete all sessions but mysession
```vim
tmux kill-session -a -t mysession
```
### Rename session
`Ctrl` + `b` `$`
### Detach from session
`Ctrl` + `b` `d`
### Detach others on the session (Maximize window by detach other clients)
```vim
:attach -d
```
### Show all sessions
```bash
tmux ls
```
`Ctrl` + `b` `s`
### Attach to last session
```bash
tmux a
```
### Attach to session with the name mysession
```bash
tmux a -t mysession
```
### Session and Window Preview
`Ctrl` + `b` `w`
### Move to previous session
`Ctrl` + `b` `(`
### Move to next session
`Ctro` + `b` `)`
## Windows
### Start a new session with the name mysession and window mywindow
```bash
tmux new -s mysession -n mywindow
```
### Create window
`Ctrl` + `b` `c`
### Rename current window
`Ctrl` + `b` `,`
### Close current window
`Ctrl` + `b` `&`
### List windows
`Ctrl` + `b` `w`
### Previous window
`Ctrl` + `b` `p`
### Next window
`Ctrl` + `b` `n`
### Switch/select window by number
`Ctrl` + `b` `0` ... `9`
### Toggle last active window
`Ctrl` + `b` `I`
### Reorder window, swap window number 2(src) and 1(dst)
```vim
:swap-window -s 2 -t 1
```
### Move current window to the left by one position
```vim
:swap-window -t -1
```
### Move window from source to target
```vim
:move-window -s src_ses:win -t target_ses:win
```
```vim
:movew -s foo:0 -t bar:9
```
```vim
:movew -s 0:0 -t 1:9
```
### Reposition window in the current session
```vim
:move-window -s src_session:src_window
```
```vim
:move -s 0:9
```
### Renumber windows to remove gap in the sequence
```vim
:movew -r
```
## Panes
### Toggle last active pane
`Ctrl` + `b` `;`
### Split the current pane with a vertical line to create a horizontal layout
```vim
:split-window -h
```
`Ctrl` + `b` `%`
### Split the current with a horizontal line to create a vertical layout
```vim
:split-window -v
```
`Ctrl` + `b` `"`
### Join two windows as panes (Merge window 2 to window 1 as panes)
```vim
:join-pane -s 2 -t 1
```
### Move pane from one window to another (Move pane 1 from window 2 to pane after 0 of window 1)
```vim
:join-pane -s 2.1 -t 1.0
```
### Move the current pane left
`Ctrl` + `b` `{`
### Move the current pane right
`Ctrl` + `b` `}`
### Switch to pane to the direction
`Ctrl` + `b` $\uparrow$
`Ctrl` + `b` $\downarrow$
`Ctrl` + `b` $\leftarrow$
`Ctrl` + `b` $\rightarrow$
### Toggle synchronize-panes(send command to all panes)
```vim
:setw synchronize-panes
```
### Toggle between pane layouts
`Ctrl` + `b` `Spacebar`
### Switch to next pane
`Ctrl` + `b` `o`
### Show pane numbers
`Ctrl` + `b` `q`
### Switch/select pane by number
`Ctrl` + `b ` `q` `0` ... `9`
### Toggle pane zoom
`Ctrl` + `b` `z`
### Convert pane into a window
`Ctrl` + `b` `!`
### Resize current pane height(holding second key is optional)
`Ctrl` + `b` + $\uparrow$
`Ctrl` + `b` `Ctrl` + $\uparrow$
`Ctrl` + `b` + $\downarrow$
`Ctrl `+ `b` `Ctrl` + $\downarrow$
### Resize current pane width(holding second key is optional)
`Ctrl` + `b` + $\rightarrow$
`Ctrl` + `b` `Ctrl` + $\rightarrow$
`Ctrl` + `b` + $\leftarrow$
`Ctrl `+ `b` `Ctrl` + $\leftarrow$
### Close current pane
`Ctrl` + `b` `x`
