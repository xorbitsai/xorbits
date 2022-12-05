/*
 * Copyright 2022 XProbe Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { ExpandLess, ExpandMore } from '@mui/icons-material'
import AccountBoxRoundedIcon from '@mui/icons-material/AccountBoxRounded'
import DeveloperBoardRoundedIcon from '@mui/icons-material/DeveloperBoardRounded'
import GridViewRoundedIcon from '@mui/icons-material/GridViewRounded'
import LabelImportantRoundedIcon from '@mui/icons-material/LabelImportantRounded'
import WorkspacesRoundedIcon from '@mui/icons-material/WorkspacesRounded'
import { Collapse } from '@mui/material'
import List from '@mui/material/List'
import ListItemButton from '@mui/material/ListItemButton'
import ListItemIcon from '@mui/material/ListItemIcon'
import ListItemText from '@mui/material/ListItemText'
import Typography from '@mui/material/Typography'
import React from 'react'
import { Link } from 'react-router-dom'

import { useStyles } from './Style'

export default function LeftMenu() {
  const classes = useStyles()
  const getHashPath = () => window.location.hash.substring(1)
  const [hash, setHash] = React.useState(getHashPath())

  window.addEventListener(
    'hashchange',
    () => {
      setHash(getHashPath())
    },
    false
  )

  const typographyText = (content) => {
    return <Typography variant="h6">{content}</Typography>
  }

  const genNodeSubMenu = (nodeRole) => {
    const match = hash.match(/^\/(supervisor|worker)\/([^/]+)/, 1)
    return (
      match &&
      nodeRole === match[1] && (
        <Collapse in={match} timeout="auto" unmountOnExit>
          <List component="div" disablePadding>
            <ListItemButton
              sx={{ pl: 4 }}
              component={Link}
              to={`/${match[1]}/${match[2]}`}
              selected={true}
            >
              <ListItemIcon>
                <LabelImportantRoundedIcon />
              </ListItemIcon>
              <ListItemText primary={match[2]} />
            </ListItemButton>
          </List>
        </Collapse>
      )
    )
  }

  const getExpend = (match) => {
    return match ? <ExpandLess /> : <ExpandMore />
  }

  const genSessionSubMenu = () => {
    const match = hash.match(/^\/session\/([^/]+)\/task/, 1)
    return (
      match && (
        <Collapse in={match} timeout="auto" unmountOnExit>
          <List component="div" disablePadding>
            <ListItemButton
              sx={{ pl: 4 }}
              component={Link}
              to={`/session/${match[1]}/task`}
              selected={true}
            >
              <ListItemIcon>
                <LabelImportantRoundedIcon />
              </ListItemIcon>
              <ListItemText primary={match[1].substring(0, 10)} />
            </ListItemButton>
          </List>
        </Collapse>
      )
    )
  }

  return (
    <List className={classes.leftMenu}>
      <div>
        <ListItemButton component={Link} to="/" selected={hash === '/'}>
          <ListItemIcon>
            <GridViewRoundedIcon />
          </ListItemIcon>
          <ListItemText primary={typographyText('Dashboard')} />
        </ListItemButton>
        <ListItemButton
          component={Link}
          to="/supervisor"
          selected={hash.startsWith('/supervisor')}
        >
          <ListItemIcon>
            <WorkspacesRoundedIcon />
          </ListItemIcon>
          <ListItemText primary={typographyText('Supervisors')} />
          {getExpend(hash.match(/^\/supervisor\/([^/]+)/, 1))}
        </ListItemButton>
        {genNodeSubMenu('supervisor')}
        <ListItemButton
          component={Link}
          to="/worker"
          selected={hash.startsWith('/worker')}
        >
          <ListItemIcon>
            <DeveloperBoardRoundedIcon />
          </ListItemIcon>
          <ListItemText primary={typographyText('Workers')} />
          {getExpend(hash.match(/^\/worker\/([^/]+)/, 1))}
        </ListItemButton>
        {genNodeSubMenu('worker')}
        <ListItemButton
          component={Link}
          to="/session"
          selected={hash === '/session'}
        >
          <ListItemIcon>
            <AccountBoxRoundedIcon />
          </ListItemIcon>
          <ListItemText primary={typographyText('Sessions')} />
          {getExpend(hash.match(/^\/session\/([^/]+)\/task/, 1))}
        </ListItemButton>
        {genSessionSubMenu()}
      </div>
    </List>
  )
}
