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

import Grid from '@mui/material/Grid'
import Paper from '@mui/material/Paper'
import Tab from '@mui/material/Tab'
import Tabs from '@mui/material/Tabs'
import PropTypes from 'prop-types'
import React from 'react'

import { useStyles } from '../Style'
import Title from '../Title'
import NodeEnvTab from './NodeEnvTab'
import NodeLogTab from './NodeLogTab'
import NodeResourceTab from './NodeResourceTab'
import NodeStackTab from './NodeStackTab'
import TabPanel from './TabPanel'

export default function WorkerDetailPage(props) {
  const classes = useStyles()
  const [value, setValue] = React.useState(0)

  const handleChange = (event, newValue) => {
    setValue(newValue)
  }

  const title_text = `${props.nodeRole.replace(/\w/, (first) =>
    first.toUpperCase()
  )}: ${props.endpoint}`
  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Title>{title_text}</Title>
      </Grid>
      <Grid item xs={12}>
        <Paper className={classes.paper}>
          <Tabs value={value} onChange={handleChange}>
            <Tab label="Environment" />
            <Tab label="Resources" />
            <Tab label="Stacks" />
            <Tab label="Logs" />
          </Tabs>
          <TabPanel value={value} index={0}>
            <NodeEnvTab endpoint={props.endpoint} />
          </TabPanel>
          <TabPanel value={value} index={1}>
            <NodeResourceTab endpoint={props.endpoint} role="1" />
          </TabPanel>
          <TabPanel value={value} index={2}>
            <NodeStackTab endpoint={props.endpoint} />
          </TabPanel>
          <TabPanel value={value} index={3}>
            <NodeLogTab endpoint={props.endpoint} role="worker" />
          </TabPanel>
        </Paper>
      </Grid>
    </Grid>
  )
}

WorkerDetailPage.propTypes = {
  nodeRole: PropTypes.string,
  endpoint: PropTypes.string,
}
