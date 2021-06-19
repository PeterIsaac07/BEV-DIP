package com.example.client_trial2;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import com.example.client_trial2.Networking.ClientSocket;
import com.example.client_trial2.Networking.Interfaces.NetEventsInterface;
import com.example.client_trial2.Networking.Models.Packet;
import com.example.client_trial2.Networking.NetConfig;

import java.util.BitSet;

public class MainActivity extends AppCompatActivity implements NetEventsInterface {

    ClientSocket cs;
    ImageView ImgView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ImgView = findViewById(R.id.iv_test);
        NetConfig nc = new NetConfig();
        cs = new ClientSocket(nc);
        cs.SetEventsListener(this);
        cs.StartAttemptToConnect();
    }

    @Override
    public void SetOnClientConnect(ClientSocket Client) {
        Log.i("Network", "Client Successfully Connected: " + Client.toString());
    }

    @Override
    public void SetOnClientSend(ClientSocket Client, Packet Message) {
        Log.i("Network: ", "Client Successfully Sent: " + Message.GetName() + " " + Message.GetContent().length + " Bytes");
    }

    @Override
    public void SetOnClientReceive(ClientSocket Client, byte[] ImgBytes) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Bitmap bmp = BitmapFactory.decodeByteArray(ImgBytes, 0, ImgBytes.length);
                ImgView.setImageBitmap(bmp);
                Log.i("Network: ", "Client Successfully Received: " + ImgBytes.length + " Bytes");
            }
        });
    }

    @Override
    public void SetOnClientDisconnect(ClientSocket Client) {
        Log.i("Network: ", "Client Successfully Disconnected");
    }

    @Override
    public void SetOnClientException(Exception Ex) {
        Log.i("Network: ", "Client Exception: " + Ex.getMessage());
    }
}