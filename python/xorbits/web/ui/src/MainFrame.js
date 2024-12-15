/*
 * Copyright 2022-2023 XProbe Inc.
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

import DescriptionIcon from '@mui/icons-material/Description'
import GithubIcon from '@mui/icons-material/GitHub'
import AppBar from '@mui/material/AppBar'
import Container from '@mui/material/Container'
import CssBaseline from '@mui/material/CssBaseline'
import Drawer from '@mui/material/Drawer'
import Link from '@mui/material/Link'
import Toolbar from '@mui/material/Toolbar'
import React from 'react'
import { HashRouter } from 'react-router-dom'

import mainImage from '../resources/xorbits.svg'
import LeftMenu from './LeftMenu'
import PageRouter from './PageRouter'
import { useStyles } from './Style'

const drawerWidth = 240

export default function MainFrame() {
  const classes = useStyles()

  return (
    <div className={classes.root}>
      <CssBaseline />
      <AppBar
        elevation={0}
        position="fixed"
        color="transparent"
        sx={{
          backdropFilter: 'blur(20px)',
          borderBottom: 1,
          borderColor: 'grey.300',
          zIndex: (theme) => theme.zIndex.drawer + 1,
        }}
      >
        <Toolbar sx={{ justifyContent: 'space-between' }}>
          <img src={mainImage} alt="logo" className={classes.logo} />
          <Toolbar sx={{ justifyContent: 'space-between' }}>
            <GithubIcon fontSize="large" />
            <Link
              align="center"
              href="https://github.com/xorbitsai/xorbits"
              underline="none"
              color="inherit"
              fontSize="large"
              sx={{ marginLeft: 1 }}
            >
              Repository
            </Link>
            <DescriptionIcon fontSize="large" sx={{ marginLeft: 3 }} />
            <Link
              align="center"
              href="https://xorbits.readthedocs.io/"
              underline="none"
              color="inherit"
              fontSize="large"
              sx={{ marginLeft: 1 }}
            >
              Documentation
            </Link>
          </Toolbar>
        </Toolbar>
      </AppBar>
      <HashRouter>
        <Drawer
          variant="permanent"
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            ['& .MuiDrawer-paper']: {
              width: drawerWidth,
              boxSizing: 'border-box',
            },
          }}
        >
          <Toolbar />
          <LeftMenu />
        </Drawer>
        <main className={classes.content}>
          <div className={classes.appBarSpacer} />
          <Container maxWidth="lg" className={classes.container}>
            <PageRouter />
          </Container>
        </main>
      </HashRouter>
    </div>
  )
}
